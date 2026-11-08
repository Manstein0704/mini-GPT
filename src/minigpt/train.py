"""Training/evaluation loop exercises; seed and checkpoint for reproducibility."""

import numpy as np
import torch
import torch.nn as nn
from minigpt.model import GPTLanguageModel
from minigpt.data import get_batch
import random
from pathlib import Path



BATCH_SIZE = 8
BLOCK_SIZE = 256
LEARNING_RATE = 3e-4
MAX_STEPS = 10000
SEED=2026
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)


def main():
    set_seed(SEED)

    train_data = np.memmap(
        "artifacts/train.bin",
        dtype=np.uint16,
        mode="r"
    )

    model = GPTLanguageModel(
        vocab_size=8000,
        embed_dim=256,
        block_size=BLOCK_SIZE,
        num_heads=4,
        num_blocks=4,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for step in range(MAX_STEPS):
        optimizer.zero_grad(set_to_none=True)
        X, Y = get_batch(
            train_data,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
        )
        X = X.to(device)
        Y = Y.to(device)

        logits, loss = model(X, Y)
        print(f"Step {step+1}/{MAX_STEPS} | Loss:{loss.item():.4f}")
        loss.backward()
        optimizer.step()

    save_path = Path("artifacts/checkpoints/minigpt_step0100.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": MAX_STEPS,
        "loss": loss.item(),
        "config": {
            "vocab_size": 8000,
            "embed_dim": 256,
            "block_size": BLOCK_SIZE,
            "num_heads": 4,
            "num_blocks": 4,
            "dropout": 0.1,
        },
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
