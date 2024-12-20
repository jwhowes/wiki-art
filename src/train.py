import torch
import os

from torch import nn
from torch.utils.data import DataLoader
from typing import Callable
from transformers import BatchEncoding, get_cosine_schedule_with_warmup

from src.config import Config
from . import accelerator


def train(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[nn.Module, BatchEncoding], torch.FloatTensor],
        config: Config
):
    opt = torch.optim.Adam(
        model.parameters(), lr=config.lr
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=config.num_epochs,
        num_training_steps=config.num_epochs * len(dataloader)
    )

    model, dataloader, opt, lr_scheduler, loss_fn = accelerator.prepare(
        model, dataloader, opt, lr_scheduler, loss_fn
    )

    model.train()
    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1} / {config.num_epochs}")

        for i, encoding in enumerate(dataloader):
            opt.zero_grad()

            loss = loss_fn(model, encoding)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)

            opt.step()
            lr_scheduler.step()

            if i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\tLoss: {loss.item():.6f}")

        torch.save(
            accelerator.get_state_dict(model),
            os.path.join(config.exp_dir, f"checkpoint_{epoch + 1:02}.pt")
        )
