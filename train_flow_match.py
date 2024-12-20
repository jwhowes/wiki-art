from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.config import Config
from src.data import ConditionalDatasetConfig, WikiArtConditionalDataset
from src.model import CrossAttentionFiLMUNetConfig, CrossAttentionFiLMUNet
from src.loss import FlowMatchLoss
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(
        CrossAttentionFiLMUNetConfig, ConditionalDatasetConfig, args.config
    )

    dataset = WikiArtConditionalDataset(
        image_size=config.dataset.image_size, text_encoder_path=config.model.text_encoder_path
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

    loss_fn = FlowMatchLoss(
        config.model.text_encoder_path, config.model.sigma_min
    )

    train(model, dataloader, loss_fn, config)
