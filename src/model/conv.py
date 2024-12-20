import torch

from torch import nn

from .util import LayerNorm2d


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

