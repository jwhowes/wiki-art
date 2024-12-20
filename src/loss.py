import torch
import torch.nn.functional as F

from torch import nn
from transformers import BatchEncoding


def mask_loss(
        model: nn.Module,
        encoding: BatchEncoding
) -> torch.FloatTensor:
    mask = encoding["mask"]
    image = encoding["image"]

    pred = model(torch.concatenate((
        mask,
        (1 - mask) * image
    ), dim=1))

    diff = F.mse_loss(pred, image, reduction="none")

    return (diff.mean(1) * mask).sum() / mask.sum()
