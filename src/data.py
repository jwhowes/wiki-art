import torch

from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from transformers import BatchEncoding

from . import accelerator
from .config import SubConfig


class WikiArtDataset(Dataset):
    @accelerator.main_process_first()
    def __init__(self, image_size=256):
        self.image_size = image_size

        self.ds = load_dataset("Artificio/WikiArt", split="train")

        transform = [
            transforms.ToTensor()
        ]
        if image_size != 256:
            transform += [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                )
            ]

        self.transform = transforms.Compose(transform + [
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def collate(images):
        return BatchEncoding({
            "image": torch.stack(images)
        })

    def __getitem__(self, idx):
        return self.transform(self.ds[idx]["image"])


class WikiArtMaskDataset(WikiArtDataset):
    def __init__(self, image_size=256, patch_size=32, mask_p=0.4):
        super(WikiArtMaskDataset, self).__init__(image_size)

        self.patch_size = patch_size
        self.mask_p = mask_p

        self.num_patches = image_size // patch_size

    @staticmethod
    def collate(batch):
        images, masks = zip(*batch)
        return BatchEncoding({
            "image": torch.stack(images),
            "mask": torch.stack(masks)
        })

    def __getitem__(self, idx):
        image = super().__getitem__(idx)

        mask = torch.zeros(
            self.num_patches * self.num_patches, self.patch_size, self.patch_size
        )
        mask[torch.rand(self.num_patches * self.num_patches) < self.mask_p] = 1.0

        mask = (
            mask
            .unflatten(0, (1, self.num_patches, self.num_patches))
            .permute(0, 1, 3, 2, 4)
            .flatten(1, 2).flatten(2)
        )

        return image, mask


class MaskDatasetConfig(SubConfig):
    def __init__(self, config=None):
        self.image_size = 256
        self.patch_size = 32
        self.mask_p = 0.4

        super().__init__(config)
