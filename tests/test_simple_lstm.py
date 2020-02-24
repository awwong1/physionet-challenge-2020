import unittest

import torch

from dataset import PhysioNet2020Dataset
from models.simple_lstm import SimpleLSTM


class SimpleLSTMTest(unittest.TestCase):
    def setUp(self):
        self.ds = PhysioNet2020Dataset(
            "Training_WFDB", fs=250, max_seq_len=5000)
        self.model = SimpleLSTM(max_seq_len=5000)

    def test_forward_backward(self):
        dl = torch.utils.data.DataLoader(
            self.ds,
            batch_size=64,
            collate_fn=PhysioNet2020Dataset.collate_fn
        )

        for x_padded, y_padded, x_lens, y_lens in dl:
            output = self.model(x_padded, x_lens)
            self.assertEqual(output.shape, (64, 9))
            break
