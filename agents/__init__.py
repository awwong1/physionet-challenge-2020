import numpy as np
import os
import random
import torch
from logging import getLogger
from shutil import copyfile
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from util.config import fetch_class, init_class


class BaseAgent:
    """Baseline agent for PyTorch machine learning experiments.
    Helper methods are indicated by leading underscore.
    """

    def __init__(self, config):
        """Agent constructor, called after config parsing
        """
        self.config = config
        self.logger = getLogger(config.get("exp_name"))

    def run(self):
        """Experiment's main logic, called by main.py.
        """
        raise NotImplementedError

    def finalize(self):
        """Called after `run` by main.py. Clean up experiment.
        """
        # raise NotImplementedError
        return

    def _setup_cuda(self, config):
        """Determine if CUDA can be used in this experiment
        """
        cpu_only = config.get("cpu_only", False)
        self.gpu_ids = config.get("gpu_ids", list(range(torch.cuda.device_count())))
        self.use_cuda = torch.cuda.is_available() and not cpu_only and self.gpu_ids
        if self.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = " ".join(map(str, self.gpu_ids))
            for gpu_id in self.gpu_ids:
                device_name = torch.cuda.get_device_name(gpu_id)
                self.logger.info("CUDA %s: %s", gpu_id, device_name)
        else:
            self.logger.info("CPU only")

    def _set_seed(self, config):
        """Set a deterministic seed, if provided in the configuration
        """
        self.seed = config.get("seed")
        if self.seed is not None:
            if self.seed is True:
                self.seed = torch.seed()
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.logger.info("Manual seed %s", self.seed)
            self.logger.warning(
                "Seed enabled with deterministic CuDNN. Training will slow down."
            )
            self.logger.warning("Restarting from checkpoints is undefined behaviour.")

    @staticmethod
    def _setup_data(config):
        """Given a PyTorch dataset configuration dictionary,
        instantiate the dataset and dataloader
        """
        ds_class = fetch_class(config["name"])
        d_transform = list(map(init_class, config.get("transform", [])))
        d_ttransform = list(map(init_class, config.get("target_transform", [])))
        ds = ds_class(
            *config.get("args", []),
            **config.get("kwargs", {}),
            transform=Compose(d_transform) if d_transform else None,
            target_transform=Compose(d_ttransform) if d_ttransform else None
        )
        dl = DataLoader(ds, **config.get("dataloader_kwargs", {}))
        return ds, dl

    @staticmethod
    def save_checkpoint(
        state,
        is_best,
        filename="checkpoint.pth.tar",
        best_filename="model_best.pth.tar",
    ):
        torch.save(state, filename)
        if is_best:
            copyfile(filename, best_filename)
