from datasets import load_dataset


if __name__ == '__main__':
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=['/scratch/high_modality/geom_train_images.jsonl'],
        split="train",
    )
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=['/scratch/high_modality/geom_train_videos.jsonl'],
        split="train",
    )
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=['/scratch/high_modality/geom_valid_images.jsonl'],
        split="train",
    )
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=['/scratch/high_modality/geom_valid_videos.jsonl'],
        split="train",
    )