import unittest

import torch

from dataset import PhysioNet2020Dataset
from models.simple_cnn import SimpleCNN


class SimpleCNNTest(unittest.TestCase):
    def setUp(self):
        self.ds = PhysioNet2020Dataset(
            "Training_WFDB", fs=200, max_seq_len=4000,
            ensure_equal_len=True, signal_last=True)
        self.model = SimpleCNN()

    def test_forward_backward(self):
        dl = torch.utils.data.DataLoader(
            self.ds,
            batch_size=64
        )

        for signal, target in dl:
            output = self.model(signal)
            self.assertEqual(output.shape, (64, 9))
            break
