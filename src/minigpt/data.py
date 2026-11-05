"""TinyStories loading, train/validation split, and token batching exercises."""

from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
import numpy as np
import torch



def load_tinystories(split="train"):
    return load_dataset("roneneldan/TinyStories", split=split)


def get_batch(token_ids, batch_size=32, block_size=256):
    if len(token_ids) <= block_size:
        raise ValueError(
            f"Not enough tokens: {len(token_ids)}"
            f"(block_size={block_size})"
        )

    
    start_idx = torch.randint(0, len(token_ids) - block_size, (batch_size,))
    X = torch.stack([
        torch.tensor(token_ids[int(i):int(i) + block_size],
                     dtype=torch.long)
                     for i in start_idx
    ])

    Y = torch.stack([
        torch.tensor(token_ids[int(i) + 1 :int(i) + block_size + 1],
                     dtype=torch.long)
                     for i in start_idx
    ])



    return X, Y



def tokenize_dataset(dataset, tokenizer):
    """
    Convert all stories into a single sequence of token IDs.

    Args:
        dataset: Hugging Face Dataset
        tokenizer: Trained BPE Tokenizer

    Returns:
        token_ids: 1D torch.Tensor
    """

    token_ids = []

    for example in dataset:
        ids = tokenizer.encode(example["text"]).ids
        token_ids.extend(ids)

    return torch.tensor(token_ids, dtype=torch.long)



def tokenize_and_save(dataset, tokenizer, output_path, batch_size=1000):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer.get_vocab_size() > 2**16:
        raise ValueError("Vocabulary too large for uint16")

    total_tokens = 0

    with open(output_path, "wb") as f:
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            texts = dataset[start:end]["text"]

            encodings = tokenizer.encode_batch(texts)
            token_ids = []
            for encoding in encodings:
                token_ids.extend(encoding.ids)
            chunk = np.asarray(token_ids, dtype=np.uint16)
            chunk.tofile(f)

            total_tokens += len(chunk)
        print(f"Saved {total_tokens:,} tokens to {output_path}")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    train_ds = load_tinystories("train")
    val_ds = load_tinystories("validation")

    tokenize_and_save(train_ds, tokenizer, "artifacts/train.bin")
    tokenize_and_save(val_ds, tokenizer, "artifacts/val.bin")

    