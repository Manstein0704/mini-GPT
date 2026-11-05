"""TinyStories loading, train/validation split, and token batching exercises."""

from datasets import load_dataset

train_ds = load_dataset("roneneldan/TinyStories", split="train")
val_ds = load_dataset("roneneldan/TinyStories", split="validation")

print(f"Train samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(train_ds.features)
print(train_ds[0].keys())

print("\nExample story:")
print(train_ds[0]["text"])
