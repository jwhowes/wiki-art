import torch
import numpy as np

from torch import nn
from transformers import CLIPTextModelWithProjection


class CLIPModel(nn.Module):
    def __init__(self, pretrained_path="openai/clip-vit-base-patch32", init_temp=0.07, min_temp=0.01):
        super(CLIPModel, self).__init__()
        self.text_model = CLIPTextModelWithProjection.from_pretrained(pretrained_path)

        self.min_t = min_temp
        self.log_t = nn.Parameter(
            torch.tensor(np.log(init_temp))
        )
