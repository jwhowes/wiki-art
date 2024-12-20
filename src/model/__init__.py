import os

from transformers import AutoConfig

from .conv import UNet, CrossAttentionFiLMUNet
from .clip import CLIPModel
from ..config import SubConfig, Config


class CLIPModelConfig(SubConfig):
    def __init__(self, config=None):
        self.pretrained_path = "openai/clip-vit-base-patch32"
        self.init_temp = 0.07
        self.min_temp = 0.01

        super().__init__(config)


class UNetConfig(SubConfig):
    def __init__(self, config=None):
        self.dims = (32, 64, 128, 256, 512)
        self.depths = (1, 1, 1, 1, 2)
        self.dropout = 0.0

        super().__init__(config)


class CrossAttentionFiLMUNetConfig(SubConfig):
    def __init__(self, config=None):
        self.d_t = 512
        self.dims = (32, 64, 128, 256, 512)
        self.n_heads = (2, 2, 2, 4, 8)
        self.depths = (1, 1, 1, 1, 2)
        self.dropout = 0.0

        self.sigma_min = 1e-4

        self.clip_exp = "clip"
        self.clip_epoch = 3

        super().__init__(config)

        clip_config = Config(CLIPModelConfig, SubConfig, os.path.join("experiments", self.clip_exp, "config.yaml"))
        text_encoder_config = AutoConfig.from_pretrained(clip_config.model.pretrained_path)
        self.d_cond = text_encoder_config.text_config.hidden_size
