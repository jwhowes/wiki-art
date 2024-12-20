import torch
import torch.nn.functional as F

from torch import nn
from transformers import CLIPVisionModelWithProjection, BatchEncoding

from .model import CLIPModel


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
    def __init__(self, text_encoder, sigma_min=1e-4):
        super(FlowMatchLoss, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.text_encoder = text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

    def forward(
            self,
            model: nn.Module,
            encoding: BatchEncoding
    ):
        B = encoding["image"].shape[0]

        cond = self.text_encoder(encoding["input_ids"], encoding["attention_mask"]).last_hidden_state
        attention_mask = torch.zeros(encoding["input_ids"].shape, device=encoding["image"].device).masked_fill(
            ~encoding["attention_mask"].to(torch.bool), float('-inf')
        )

        t = torch.rand(B, device=encoding["image"].device).view(B, 1, 1, 1)
        x_0 = torch.randn_like(encoding["image"])
        x_t = (1 - self.sigma_offset * t) * x_0 + t * encoding["image"]

        pred_flow = model(x_t, t, cond, attention_mask)

        return F.mse_loss(pred_flow, encoding["image"] - self.sigma_offset * x_0)


class CLIPLoss(nn.Module):
    def __init__(self, pretrained_path="openai/clip-vit-base-patch32"):
        super(CLIPLoss, self).__init__()

        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_path)
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)

    def forward(
            self,
            model: CLIPModel,
            encoding: BatchEncoding
    ):
        image_features = self.vision_model(encoding["image"]).image_embeds
        text_features = model.text_model(encoding["input_ids"], encoding["attention_mask"]).text_embeds

        logits = (text_features @ image_features.T) / model.log_t.exp().clamp(min=model.min_t)
        tgt = torch.arange(logits.shape[0], device=logits.device)

        return F.cross_entropy(logits, tgt)
