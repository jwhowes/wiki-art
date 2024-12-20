import torch
import torch.nn.functional as F

from torch import nn
from transformers import BatchEncoding, CLIPTextModel


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


class FlowMatchLoss(nn.Module):
    def __init__(self, text_encoder_path="openai/clip-vit-base-patch32", sigma_min=1e-4):
        super(FlowMatchLoss, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

    def forward(
            self,
            model: nn.Module,
            encoding: BatchEncoding
    ):
        B = encoding["image"].shape[0]

        cond = self.text_encoder(**encoding["cond"]).last_hidden_state
        attention_mask = torch.zeros(encoding["cond"]["input_ids"].shape, device=encoding["image"].device).masked_fill(
            ~encoding["cond"]["attention_mask"].to(torch.bool), float('-inf')
        )

        t = torch.rand(B, device=encoding["image"].device).view(B, 1, 1, 1)
        x_0 = torch.randn_like(encoding["image"])
        x_t = (1 - self.sigma_offset * t) * x_0 + t * encoding["image"]

        pred_flow = model(x_t, t, cond, attention_mask)

        return F.mse_loss(pred_flow, encoding["image"] - self.sigma_offset * x_0)
