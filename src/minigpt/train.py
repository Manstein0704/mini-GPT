"""Training/evaluation loop exercises; seed and checkpoint for reproducibility."""

import numpy as np
import torch
import torch.nn as nn
from minigpt.model import GPTLanguageModel
from minigpt.data import get_batch
import random
from pathlib import Path
import csv 
from pathlib import Path
from minigpt.evaluate import estimate_loss



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

    val_data = np.memmap(
        "artifacts/val.bin",
        dtype=np.uint16,
        mode="r"
    )

    EVALINTERVAL = 100
    EVAL_ITERS = 20

    log_path = Path("artifacts/logs/loss.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_config = {
        "vocab_size": 8000,
        "embed_dim": 512,
        "block_size": BLOCK_SIZE,
        "num_heads": 8,
        "num_blocks": 8,
        "dropout": 0.1,
    }

    model = GPTLanguageModel(**model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])

        for step in range(MAX_STEPS):
            # Training
            model.train()

            X, Y = get_batch(
                train_data,
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE
            )

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, loss = model(X, Y)

            loss.backward()
            optimizer.step()

            # 現在のTraining Lossを表示
            if (step + 1) % 10 == 0:
                print(
                    f"Step {step+1}/{MAX_STEPS} | "
                    f"Batch Loss: {loss.item():.4f}"
                )

            # 100ステップごとに評価
            if (step + 1) % EVALINTERVAL == 0:
                losses = estimate_loss(
                    model=model,
                    train_data=train_data,
                    val_data=val_data,
                    batch_size=BATCH_SIZE,
                    block_size=BLOCK_SIZE,
                    eval_iters=EVAL_ITERS,
                    device=device
                )

                train_loss = losses["train"]
                val_loss = losses["val"]

                # CSVに記録
                writer.writerow([
                    step + 1,
                    train_loss,
                    val_loss
                ])
                f.flush()

                print(
                    f"[Evaluation] Step {step+1}/{MAX_STEPS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

    save_path = Path(f"artifacts/checkpoints/minigpt_step{MAX_STEPS:06d}.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": MAX_STEPS,
        "loss": loss.item(),
        "config": model_config,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "seed": SEED,
        },
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
