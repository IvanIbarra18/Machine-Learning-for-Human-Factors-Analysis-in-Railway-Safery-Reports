
import numpy as np
import torch
from tqdm import tqdm

from utilis.metrics import average_precision_per_event


@torch.no_grad()
def evaluate(model,dataloader,criterion,device,use_amp=False,amp_dtype=torch.float32,desc="Validation",):
    """Run a multi-label classifier over a DataLoader.

    Returns the average loss, each event's probability for every report, the 0/1
    targets, and average precision per event plus their mean.
    """

    model.eval()

    total_loss = 0.0
    samples_seen = 0
    all_targets = []
    all_probabilities = []

    # disable=None hides the bar when output is not a terminal (for example a log file).
    with tqdm(total=len(dataloader.dataset),desc=desc,unit="sample",dynamic_ncols=True,disable=None,) as pbar:
        for x_cont, x_cat, targets in dataloader:
            x_cont = x_cont.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass with optional automatic mixed precision.
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp,):
                logits = model(x_cont, x_cat)

            # Calculate loss in FP32 for numerical stability.
            logits = logits.float()
            loss = criterion(logits, targets)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            samples_seen += batch_size

            all_targets.append(targets.cpu())
            all_probabilities.append(torch.sigmoid(logits).cpu())

            pbar.update(batch_size)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    if samples_seen == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    targets = torch.cat(all_targets).numpy().astype(int)
    probabilities = torch.cat(all_probabilities).numpy()
    ap = average_precision_per_event(targets, probabilities)

    return {
        "loss": total_loss / samples_seen,
        "probabilities": probabilities,
        "targets": targets,
        "ap": ap,
        "mean_ap": float(np.mean(ap)),
    }
