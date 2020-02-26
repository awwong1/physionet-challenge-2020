import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):
    def __init__(self, data_dir):
        """Helper dataset class to consume the preprocessed signals
        """
        super(ProcessedDataset, self).__init__()
        self.samples = sorted(glob(os.path.join(data_dir, "*.npz")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        sig = torch.FloatTensor(data["signal"])
        target = torch.FloatTensor(data["target"])
        return sig, target
