import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ...extras.compute_metrics import compute_metrics_by_data_source

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
    def evaluate(
            self,
            eval_dataset: Optional["Dataset"] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation with hierarchical metrics by data source and dataset.
        Processes the dataset sequentially to ensure correct mapping between
        predictions and their metadata.

        Args:
            eval_dataset: The evaluation dataset.
            ignore_keys: Keys to ignore when gathering predictions.
            metric_key_prefix: The prefix for metric keys in the returned dictionary.
            **gen_kwargs: Generation kwargs to pass to model.generate().

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        # Set up evaluation dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Create dataloader for evaluation
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Initialize lists to store predictions, ground truths, and metadata
        all_preds = []
        all_labels = []
        all_data_sources = []
        all_datasets = []

        # Set model to evaluation mode
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        # Put model on eval mode
        model.eval()

        # If only the model is provided to accelerator.prepare() (not the optimizer, etc),
        # it may lead to unexpected behaviors (e.g., CPU instead of GPU usage).
        # We verify if the model is on the correct device.
        if self.args.device.type != "cuda" and torch.cuda.is_available():
            self.logger.warning_once(
                "The model is not on the expected device: "
                f"({self.args.device.type} vs. cuda). "
                "It might lead to degraded performance."
            )

        # Initialize metrics
        metrics = {}

        # Process each batch in the dataloader
        for step, inputs in enumerate(eval_dataloader):
            # Extract original samples from batch for metadata extraction
            batch_size = len(inputs["input_ids"]) if isinstance(inputs, dict) else inputs[0].shape[0]

            # Extract data sources and datasets from this batch
            batch_data_sources = []
            batch_datasets = []

            # Process each example in the batch to extract metadata
            for i in range(batch_size):
                # Get the original example from the dataset
                example_idx = step * eval_dataloader.batch_size + i
                if example_idx >= len(eval_dataset):
                    continue  # Skip if we're beyond the dataset size (last batch might be incomplete)

                example = eval_dataset[example_idx]

                # Extract row_dict from example
                row_dict = example
                if isinstance(example, dict) and "json" in example:
                    try:
                        row_dict = json.loads(example["json"])
                    except:
                        row_dict = example

                # Extract data source and dataset based on the provided logic
                # First check if we have the original paths (from the enhanced converter)
                if "_original_image_paths" in row_dict and row_dict["_original_image_paths"] and len(
                        row_dict["_original_image_paths"]) > 0:
                    path = row_dict["_original_image_paths"][0]
                    if isinstance(path, str) and "/" in path:
                        parts = path.split("/")
                        if len(parts) >= 2:
                            data_source = parts[0]
                            dataset = parts[1]
                        else:
                            data_source = "unknown"
                            dataset = "image"
                    else:
                        data_source = "unknown"
                        dataset = "image"
                elif "_original_video_paths" in row_dict and row_dict["_original_video_paths"] and len(
                        row_dict["_original_video_paths"]) > 0:
                    path = row_dict["_original_video_paths"][0]
                    if isinstance(path, str) and "/" in path:
                        parts = path.split("/")
                        if len(parts) >= 2:
                            data_source = parts[0]
                            dataset = parts[1]
                        else:
                            data_source = "unknown"
                            dataset = "video"
                    else:
                        data_source = "unknown"
                        dataset = "video"
                # Fall back to the processed paths if original paths aren't available
                elif "_images" in row_dict and row_dict["_images"] is not None and len(row_dict["_images"]) > 0:
                    image_path = row_dict["_images"][0]
                    path_parts = image_path.split("/")
                    # Look for parts that might correspond to data source and dataset
                    # This is less reliable than using original paths
                    for i in range(len(path_parts) - 1):
                        if i + 1 < len(path_parts):
                            data_source = path_parts[i]
                            dataset = path_parts[i + 1]
                            break
                    else:
                        data_source = "unknown"
                        dataset = "image"
                elif "_videos" in row_dict and row_dict["_videos"] is not None and len(row_dict["_videos"]) > 0:
                    video_path = row_dict["_videos"][0]
                    path_parts = video_path.split("/")
                    # Look for parts that might correspond to data source and dataset
                    # This is less reliable than using original paths
                    for i in range(len(path_parts) - 1):
                        if i + 1 < len(path_parts):
                            data_source = path_parts[i]
                            dataset = path_parts[i + 1]
                            break
                    else:
                        data_source = "unknown"
                        dataset = "video"
                else:
                    data_source = "text"
                    dataset = "text"

                # Add debugging for the first few examples
                if example_idx < 5:  # Only log for the first 5 examples to avoid flooding logs
                    logger.info(f"Example {example_idx}: data_source={data_source}, dataset={dataset}")
                    if "_images" in row_dict:
                        logger.info(f"  _images: {row_dict.get('_images')}")
                    if "_original_image_paths" in row_dict:
                        logger.info(f"  _original_image_paths: {row_dict.get('_original_image_paths')}")
                    if "_videos" in row_dict:
                        logger.info(f"  _videos: {row_dict.get('_videos')}")
                    if "_original_video_paths" in row_dict:
                        logger.info(f"  _original_video_paths: {row_dict.get('_original_video_paths')}")

                batch_data_sources.append(data_source)
                batch_datasets.append(dataset)

            # Move inputs to appropriate device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)

            # Don't pass labels to model when generating
            if self.args.predict_with_generate:
                input_ids = inputs["input_ids"] if "input_ids" in inputs else None
                labels = inputs.pop("labels") if "labels" in inputs else None
                attention_mask = inputs.get("attention_mask", None)

                # Generate predictions
                with torch.no_grad():
                    generated_tokens = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs
                    )

                    # Process the generated tokens
                    if generated_tokens is not None:
                        generated_tokens[:, :input_ids.size(-1)] = self.processing_class.pad_token_id
                        generated_tokens = generated_tokens.contiguous()

                # Decode predictions and labels
                if labels is not None:
                    labels = labels.cpu().numpy()
                    labels = np.where(
                        labels != IGNORE_INDEX, labels, self.processing_class.pad_token_id
                    )
                    decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)
                    all_labels.extend(decoded_labels)

                if generated_tokens is not None:
                    generated_tokens = generated_tokens.cpu().numpy()
                    generated_tokens = np.where(
                        generated_tokens != IGNORE_INDEX, generated_tokens, self.processing_class.pad_token_id
                    )
                    decoded_preds = self.processing_class.batch_decode(generated_tokens, skip_special_tokens=True)
                    all_preds.extend(decoded_preds)

                # Store metadata
                all_data_sources.extend(batch_data_sources)
                all_datasets.extend(batch_datasets)

        # Compute hierarchical metrics using collected predictions and metadata
        if len(all_preds) > 0 and len(all_labels) > 0:
            hierarchical_metrics = compute_metrics_by_data_source(
                predictions=all_preds,
                ground_truths=all_labels,
                data_sources=all_data_sources,
                datasets=all_datasets
            )

            # Add hierarchical metrics to the final metrics dictionary
            metrics.update(hierarchical_metrics)

            # Log metrics
            self.log(metrics)

            # Save predictions to file for later analysis
            if self.is_world_process_zero():
                output_prediction_file = os.path.join(self.args.output_dir, "hierarchical_predictions.jsonl")
                logger.info_rank0(f"Saving hierarchical prediction results to {output_prediction_file}")

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