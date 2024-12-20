import os
import torch

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.config import Config
from src.data import ConditionalDatasetConfig, WikiArtConditionalDataset
from src.model import CrossAttentionFiLMUNetConfig, CLIPModelConfig, CLIPModel, CrossAttentionFiLMUNet
from src.loss import FlowMatchLoss
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(
        CrossAttentionFiLMUNetConfig, ConditionalDatasetConfig, args.config
    )

    clip_config = Config(
        CLIPModelConfig, ConditionalDatasetConfig, os.path.join("experiments", config.model.clip_exp, "config.yaml")
    )

    dataset = WikiArtConditionalDataset(
        image_size=config.dataset.image_size,
        p_uncond=config.dataset.p_uncond, text_encoder_path=clip_config.model.pretrained_path
    )

    model = CrossAttentionFiLMUNet(
        in_channels=3,
        d_cond=config.model.d_cond,
        d_t=config.model.d_t,
        n_heads=config.model.n_heads,
        dims=config.model.dims,
        depths=config.model.depths,
        dropout=config.model.dropout
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate
    )

    clip_model = CLIPModel(
        pretrained_path=clip_config.model.pretrained_path, init_temp=clip_config.model.init_temp,
        min_temp=clip_config.model.min_temp
    )
    ckpt = torch.load(
        os.path.join("experiments", config.model.clip_exp, f"checkpoint_{config.model.clip_epoch:02}.pt"),
        weights_only=True, map_location="cpu"
    )
    clip_model.load_state_dict(ckpt)

    loss_fn = FlowMatchLoss(
        clip_model.text_model.text_model, config.model.sigma_min
    )

    train(model, dataloader, loss_fn, config)
