from datasets import load_dataset


if __name__ == '__main__':
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=['/mnt/8T/high_modality/geom_train_images.jsonl'],
        split="train",
    )