import yaml
import os

from torch import nn
from torch.utils.data import Dataset
from typing import Optional, Dict, Type
from abc import ABC, abstractmethod


class SubConfig:
    def __init__(self, config: Optional[Dict] = None):
        if config is not None:
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)


class Config(SubConfig):
    def unknown_tag(self, loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            constructor = loader.__class__.construct_scalar
        elif isinstance(node, yaml.SequenceNode):
            constructor = loader.__class__.construct_sequence
        elif isinstance(node, yaml.MappingNode):
            constructor = loader.__class__.construct_mapping
        else:
            raise NotImplementedError

        data = constructor(loader, node)

        return data

    def __init__(
            self,
            model_class: Type[SubConfig],
            dataset_class: Type[SubConfig],
            config_path: str
    ):
        yaml.add_multi_constructor('!', self.unknown_tag)
        yaml.add_multi_constructor('tag:', self.unknown_tag)

        self.lr = 3e-4
        self.batch_size = 16
        self.num_warmup_steps = 1000
        self.num_epochs = 5
        self.clip_grad = 3.0
        self.log_interval = 100

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super().__init__(config)
        self.lr = float(self.lr)

        if config:
            self.model = model_class(config["model"] if "model" in config else None)
            self.dataset = dataset_class(config["dataset"] if "dataset" in config else None)
        else:
            self.model = model_class()
            self.dataset = dataset_class()

        self.exp_name = os.path.splitext(os.path.basename(config_path))[0]
        self.exp_dir = os.path.join("experiments", self.exp_name)
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

        with open(os.path.join(self.exp_dir, "config.yaml"), "w+") as f:
            yaml.dump(self, f)
