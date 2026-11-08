
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss(csv_path, output_path):
    steps = []
    train_losses = []
    val_losses = []

    # Read CSV
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    # Plot learning curves
    plt.figure(figsize=(8, 5))

    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Validation Loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Mini-GPT Learning Curves")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Plot saved: {output_path}")


if __name__ == "__main__":
    plot_loss(
        "artifacts/logs/loss.csv",
        "artifacts/logs/loss.png"
    )
