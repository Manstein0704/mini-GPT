"""TinyStories loading, train/validation split, and token batching exercises."""

from datasets import load_dataset
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
        token_ids[i:i+block_size] for i in start_idx
    ])

    Y = torch.stack([
        token_ids[i+1:i+block_size+1] for i in start_idx
    ])

    return X, Y




if __name__ == "__main__":
    train_ds = load_tinystories("train")
    val_ds = load_tinystories("validation")

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(train_ds[0]["text"])


    token_ids = torch.arange(1000)
    X, Y = get_batch(token_ids, batch_size=4, block_size=8)
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    print(f"X:{X}")
    print(f"Y:{Y}")
    assert X.shape == (4, 8)
    assert Y.shape == (4, 8)
    assert torch.equal(X[:, 1:], Y[:, :-1])