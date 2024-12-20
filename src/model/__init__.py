from transformers import AutoConfig

from .conv import UNet, CrossAttentionFiLMUNet
from ..config import SubConfig


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

        self.text_encoder_path = "openai/clip-vit-base-patch32"

        super().__init__(config)

        text_encoder_config = AutoConfig.from_pretrained(self.text_encoder_path)
        self.d_cond = text_encoder_config.text_config.hidden_size
