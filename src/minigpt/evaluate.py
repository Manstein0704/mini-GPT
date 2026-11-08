import torch
from minigpt.data import get_batch


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, batch_size=8, block_size=256, eval_iters=20):
    model.eval()
    results = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(eval_iters):
            X, Y = get_batch(data, batch_size=batch_size, block_size=block_size)

            X = X.to(device)
            Y = Y.to(device)

            _, loss = model(X, Y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results

