"""TinyStories loading, train/validation split, and token batching exercises."""

from datasets import load_dataset


def load_tinystories(split="train"):
    return load_dataset("roneneldan/TinyStories", split=split)

if __name__ == "__main__":
    train_ds = load_tinystories("train")
    val_ds = load_tinystories("validation")

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(train_ds[0]["text"])