from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.config import Config
from src.data import MaskDatasetConfig, WikiArtMaskDataset
from src.model import UNetConfig, UNet
from src.loss import mask_loss
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(
        UNetConfig, MaskDatasetConfig, args.config
    )

    dataset = WikiArtMaskDataset(
        image_size=config.dataset.image_size, patch_size=config.dataset.patch_size, mask_p=config.dataset.mask_p
    )

    model = UNet(
        in_channels=4, out_channels=3,
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

    train(model, dataloader, mask_loss, config)
