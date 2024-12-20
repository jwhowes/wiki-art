from .conv import UNet
from ..config import SubConfig


class UNetConfig(SubConfig):
    def __init__(self, config=None):
        self.dims = (32, 64, 128, 256, 512)
        self.depths = (1, 1, 1, 1, 2)
        self.dropout = 0.0

        super().__init__(config)
