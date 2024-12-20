import torch

from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


class FlowMatchSampler(nn.Module):
    def __init__(self, flow_model, text_encoder_path="openai/clip-vit-base-patch32", image_channels=3, image_size=256):
        super(FlowMatchSampler, self).__init__()
        self.image_channels = image_channels
        self.image_size = image_size

        self.flow_model = flow_model
        self.flow_model.eval()
        self.flow_model.requires_grad_(False)

        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)

    @torch.inference_mode()
    def pred_flow(self, x_t, t, cond, attention_mask, guidance_scale=2.5):
        if guidance_scale > 1.0:
            pred_uncond, pred_cond = self.flow_model(
                torch.concat([x_t] * 2),
                torch.concat([t] * 2),
                cond,
                attention_mask
            ).chunk(2)
            return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            return self.flow_model(x_t, t, cond, attention_mask)

    @torch.inference_mode()
    def forward(self, text, num_steps=200, guidance_scale=2.5, step="euler"):
        dt = 1 / num_steps

        if guidance_scale > 1.0:
            text = ["" for _ in range(len(text))] + text

        text_encoding = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.text_encoder.device)
        B = text_encoding["input_ids"].shape[0]

        cond = self.text_encoder(**text_encoding)
        attention_mask = torch.zeros(
            text_encoding["input_ids"].shape, device=self.text_encoder.device
        ).masked_fill(
            ~text_encoding["attention_mask"].to(torch.bool), float('-inf')
        )

        x_t = torch.randn(B, self.image_channels, self.image_size, self.image_size)

        ts = torch.linspace(0, 1, num_steps).unsqueeze(1).expand(-1, B)
        for i in tqdm(range(num_steps)):
            pred_flow = self.pred_flow(x_t, ts[i], cond, attention_mask, guidance_scale)

            if step == "euler":
                x_t = x_t + dt * pred_flow
            elif step == "midpoint":
                x_t = x_t + dt * self.pred_flow(
                    x_t + 0.5 * dt * pred_flow, ts[i] + 0.5 * dt, cond, attention_mask, guidance_scale
                )
            elif step == "heun":
                if i == num_steps - 1:
                    x_t = x_t + dt * pred_flow
                else:
                    x_t = x_t + 0.5 * dt * (pred_flow + self.pred_flow(
                        x_t + dt * pred_flow, ts[i + 1], cond, attention_mask, guidance_scale
                    ))
            else:
                raise NotImplementedError
