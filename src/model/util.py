import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from math import sqrt


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FiLM2d(nn.Module):
    def __init__(self, d_model, d_t, *norm_args, **norm_kwargs):
        super(FiLM2d, self).__init__()
        self.norm = LayerNorm2d(d_model, *norm_args, elementwise_affine=False, **norm_kwargs)
        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]

        return self.gamma(t).view(B, -1, 1, 1) * self.norm(x) + self.beta(t).view(B, -1, 1, 1)


class CrossAttention2d(nn.Module):
    def __init__(self, d_model, d_cond, n_heads):
        super(CrossAttention2d, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_cond, d_model, bias=False)
        self.W_v = nn.Linear(d_cond, d_model, bias=False)

        self.W_o = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, image, cond, attention_mask=None):
        L_cond = cond.shape[1]
        B, _, H, W = image.shape

        q = rearrange(
            self.W_q(rearrange(
                image, "b d h w -> b (h w) d"
            )), "b l (n d) -> b n l d", n=self.n_heads
        )
        k = rearrange(self.W_k(cond), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(cond), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask.view(B, 1, -1, L_cond)

        return self.W_o(
            rearrange(F.softmax(attn, dim=-1) @ v, "b n (h w) d -> b (n d) h w", h=H, w=W)
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model, base=1e5):
        super(SinusoidalPosEmb, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.float().view(-1, 1) * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).view(B, -1)
