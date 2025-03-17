import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
import wandb
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ...extras.compute_metrics import compute_metrics_by_data_source

from datasets import concatenate_datasets
from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, has_length
import tqdm

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
            self,
            finetuning_args: "FinetuningArguments",
            processor: Optional["ProcessorMixin"],
            gen_kwargs: Optional[dict[str, Any]] = None,
            **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")


        self.dataset_dir = kwargs.pop("dataset_dir", "/")
        print(f"dataset_dir: {self.dataset_dir}")
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
            self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
            self,
            model: "torch.nn.Module",
            inputs: dict[str, Union["torch.Tensor", Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if "paths" in inputs:
            inputs.pop("paths")
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    @override
    def evaluate(
            self,
            eval_dataset: Optional["Dataset"] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation with distributed support by using Accelerator's primitives.
        """
        # Set up evaluation dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Create dataloader for evaluation
        if isinstance(eval_dataset, dict):
            combined_dataset = concatenate_datasets(list(eval_dataset.values()))
            eval_dataloader = self.get_eval_dataloader(combined_dataset)
        else:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # This is a much simpler approach using the accelerator's gather functionality
        # Initialize containers for results
        all_preds = []
        all_labels = []
        all_data_sources = []
        all_datasets = []

        # Prepare the model
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        # Log info
        logger.info("\n***** Running evaluation *****")
        logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        # Evaluation loop
        for step, inputs in enumerate(tqdm.tqdm(eval_dataloader, desc="Evaluation")):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)

            # Extract metadata
            batch_size = len(inputs["input_ids"])
            batch_data_sources = []
            batch_datasets = []

            for i in range(batch_size):
                if "paths" in inputs and inputs["paths"] and len(inputs["paths"]) > 0:
                    path = inputs['paths'][i][0]
                    base_path = self.dataset_dir
                    path = os.path.relpath(path, base_path)
                    parts = path.split("/")
                    assert len(parts) >= 2
                    data_source = parts[0]
                    dataset = parts[1]
                else:
                    data_source = "unknown"
                    dataset = "unknown"

                batch_data_sources.append(data_source)
                batch_datasets.append(dataset)

            # Generate predictions
            with torch.no_grad():
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)
                labels = inputs.pop("labels") if "labels" in inputs else None

                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

                # Process generated tokens
                if generated_tokens is not None:
                    generated_tokens[:, :input_ids.size(-1)] = self.processing_class.pad_token_id
                    generated_tokens = generated_tokens.contiguous()

                # Process labels
                if labels is not None:
                    # Convert labels to CPU before numpy conversion
                    labels = labels.cpu().numpy()
                    labels = np.where(
                        labels != IGNORE_INDEX, labels, self.processing_class.pad_token_id
                    )
                    decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)

                    # Use accelerator to gather across processes - ensuring we get all labels
                    gathered_labels = self.accelerator.gather_for_metrics(decoded_labels)
                    all_labels.extend(gathered_labels)

                # Process predictions
                if generated_tokens is not None:
                    # Convert to CPU before numpy conversion
                    generated_tokens = generated_tokens.cpu().numpy()
                    generated_tokens = np.where(
                        generated_tokens != IGNORE_INDEX, generated_tokens, self.processing_class.pad_token_id
                    )
                    decoded_preds = self.processing_class.batch_decode(generated_tokens, skip_special_tokens=True)

                    # Use accelerator to gather across processes
                    gathered_preds = self.accelerator.gather_for_metrics(decoded_preds)
                    all_preds.extend(gathered_preds)

                # Gather metadata
                gathered_sources = self.accelerator.gather_for_metrics(batch_data_sources)
                gathered_datasets = self.accelerator.gather_for_metrics(batch_datasets)

                all_data_sources.extend(gathered_sources)
                all_datasets.extend(gathered_datasets)

        # Compute metrics only on the main process after gathering all data
        metrics = {}
        if len(all_preds) > 0 and len(all_labels) > 0:
            # Compute hierarchical metrics
            hierarchical_metrics = compute_metrics_by_data_source(
                predictions=all_preds,
                ground_truths=all_labels,
                data_sources=all_data_sources,
                datasets=all_datasets
            )

            metrics.update(hierarchical_metrics)

            # Log metrics
            self.log(metrics)
            if self.is_world_process_zero():
                wandb.log(metrics, step=self.state.global_step)

                # Save predictions to file
                output_prediction_file = os.path.join(self.args.output_dir, "hierarchical_predictions.jsonl")
                logger.info(f"Saving hierarchical prediction results to {output_prediction_file}")

                with open(output_prediction_file, "w", encoding="utf-8") as f:
                    for pred, label, source, dataset in zip(all_preds, all_labels, all_data_sources, all_datasets):
                        f.write(json.dumps({
                            "predict": pred,
                            "label": label,
                            "data_source": source,
                            "dataset": dataset
                        }, ensure_ascii=False) + "\n")

        return metrics

    def save_predictions(
            self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    def save_predictions_with_metrics(
            self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        """Save model predictions to `output_dir` with additional metrics.

        An extended version of save_predictions that includes metrics for each prediction.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions_with_metrics.jsonl")
        logger.info_rank0(f"Saving prediction results with metrics to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        # For each prediction, calculate metrics
        from .metrics import medical_compute_score

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f1_score, metrics = medical_compute_score(pred, label)

                # Get row_dict to extract data source and dataset
                row_dict = {}
                try:
                    row_dict = json.loads(text)
                except:
                    pass

                # Extract data source and dataset
                if "images" in row_dict and len(row_dict["images"]) > 0:
                    data_source = row_dict["images"][0].split("/")[0]
                    dataset = row_dict["images"][0].split("/")[1]
                elif "videos" in row_dict and len(row_dict["videos"]) > 0:
                    data_source = row_dict["videos"][0].split("/")[0]
                    dataset = row_dict["videos"][0].split("/")[1]
                else:
                    data_source = "text"
                    dataset = "text"

                f.write(json.dumps({
                    "prompt": text,
                    "predict": pred,
                    "label": label,
                    "f1_score": f1_score,
                    "metrics": metrics,
                    "data_source": data_source,
                    "dataset": dataset
                }, ensure_ascii=False) + "\n")