import torch

from torch import nn

from .util import LayerNorm2d, FiLM2d, CrossAttention2d, SinusoidalPosEmb


class Block(nn.Module):
    def __init__(self, d_model, hidden_size=None, dropout=0.0, norm_eps=1e-6):
        super(Block, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.module = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            LayerNorm2d(d_model, eps=norm_eps),
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.module(x)


class FiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, hidden_size=None, dropout=0.0, norm_eps=1e-6):
        super(FiLMBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = FiLM2d(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x, t):
        return x + self.ffn(
            self.norm(
                self.dwconv(x), t
            )
        )


class CrossAttentionFiLMBlock(nn.Module):
    def __init__(self, d_model, d_cond, n_heads, d_t, hidden_size=None, dropout=0.0, norm_eps=1e-6):
        super(CrossAttentionFiLMBlock, self).__init__()
        self.attn = CrossAttention2d(d_model, d_cond, n_heads)
        self.attn_norm = FiLM2d(d_model, d_t, eps=norm_eps)

        self.block = FiLMBlock(d_model, d_t, hidden_size, dropout, norm_eps)

    def forward(self, x, t, cond, attention_mask=None):
        x = x + self.attn(
            self.attn_norm(x, t), cond, attention_mask
        )

        return self.block(x, t)


class CrossAttentionFiLMUNet(nn.Module):
    def __init__(
            self, in_channels, out_channels=None, d_cond=512, d_t=512, n_heads=(2, 2, 2, 4, 8),
            dims=(32, 64, 128, 256, 512), depths=(1, 1, 1, 1, 2),
            dropout=0.0
    ):
        super(CrossAttentionFiLMUNet, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            down_blocks = nn.ModuleList([
                CrossAttentionFiLMBlock(dims[i], d_cond, n_heads[i], d_t, dropout=dropout)
            ])
            for j in range(depths[i] - 1):
                down_blocks.append(FiLMBlock(
                    dims[i], d_t, dropout=dropout
                ))
            self.down_path.append(down_blocks)
            self.down_samples.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        self.mid_blocks = nn.ModuleList([
            CrossAttentionFiLMBlock(dims[-1], d_cond, n_heads[-1], d_t, dropout=dropout)
        ])
        for i in range(depths[-1] - 1):
            self.mid_blocks.append(FiLMBlock(dims[-1], d_t, dropout=dropout))

        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_path = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                dims[i + 1], dims[i], kernel_size=2, stride=2
            ))
            self.up_combines.append(nn.Conv2d(
                2 * dims[i], dims[i], kernel_size=1
            ))

            up_blocks = nn.ModuleList([
                CrossAttentionFiLMBlock(dims[i], d_cond, n_heads[i], d_t, dropout=dropout)
            ])
            for j in range(depths[i] - 1):
                up_blocks.append(FiLMBlock(
                    dims[i], d_t, dropout=dropout
                ))
            self.up_path.append(up_blocks)

        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=7, padding=3)

    def forward(self, x, t, cond, attention_mask=None):
        x = self.stem(x)

        acts = []
        for down_blocks, down_sample in zip(self.down_path, self.down_samples):
            x = down_blocks[0](x, t, cond, attention_mask)
            for block in down_blocks[1:]:
                x = block(x, t)

            acts.append(x)
            x = down_sample(x)

        x = self.mid_blocks[0](x, t, cond, attention_mask)
        for block in self.mid_blocks[1:]:
            x = block(x, t)

        for up_blocks, up_sample, up_combine, act in zip(self.up_path, self.up_samples, self.up_combines, acts[::-1]):
            x = up_combine(torch.concatenate((
                up_sample(x),
                act
            ), dim=1))

            x = up_blocks[0](x, t, cond, attention_mask)
            for block in up_blocks[1:]:
                x = block(x, t)

        return self.head(x)


class UNet(nn.Module):
    def __init__(
            self, in_channels, out_channels=None, dims=(32, 64, 128, 256, 512), depths=(1, 1, 1, 1, 2), dropout=0.0
    ):
        super(UNet, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.Sequential(*[
                Block(dims[i], dropout=dropout) for _ in range(depths[i])
            ]))
            self.down_samples.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        self.mid_blocks = nn.Sequential(*[
            Block(dims[-1], dropout=dropout) for _ in range(depths[-1])
        ])

        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_path = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                dims[i + 1], dims[i], kernel_size=2, stride=2
            ))
            self.up_combines.append(nn.Conv2d(
                2 * dims[i], dims[i], kernel_size=1
            ))
            self.up_path.append(nn.Sequential(*[
                Block(dims[i], dropout=dropout) for _ in range(depths[-1])
            ]))

        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.stem(x)

        acts = []
        for blocks, sample in zip(self.down_path, self.down_samples):
            x = blocks(x)
            acts.append(x)
            x = sample(x)

        x = self.mid_blocks(x)

        for act, blocks, sample, combine in zip(acts[::-1], self.up_path, self.up_samples, self.up_combines):
            x = combine(torch.concatenate((
                sample(x),
                act
            ), dim=1))

            x = blocks(x)

        return self.head(x)

