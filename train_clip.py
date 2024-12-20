from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.config import Config
from src.data import ConditionalDatasetConfig, WikiArtConditionalDataset
from src.model import CLIPModelConfig, CLIPModel
from src.loss import CLIPLoss
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(
        CLIPModelConfig, ConditionalDatasetConfig, args.config
    )

    dataset = WikiArtConditionalDataset(
        image_size=config.dataset.image_size, p_uncond=config.dataset.p_uncond, text_encoder_path=config.model.pretrained_path
    )

    model = CLIPModel(
        pretrained_path=config.model.pretrained_path, init_temp=config.model.init_temp, min_temp=config.model.min_temp
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate
    )

    loss_fn = CLIPLoss(pretrained_path=config.model.pretrained_path)

    train(model, dataloader, loss_fn, config)
