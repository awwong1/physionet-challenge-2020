import os
import unittest

from torch.utils.data import DataLoader
from datasets import PhysioNet2020Dataset
from models.simple_lstm import SimpleLSTM


class SimpleLSTMTest(unittest.TestCase):
    def test_forward(self):
        ds = PhysioNet2020Dataset(
            "Training_WFDB", max_seq_len=4000, records=("A0001"), proc=0
        )
        m = SimpleLSTM()
        dl = DataLoader(
            ds,
            batch_size=64,
            num_workers=0,
            collate_fn=PhysioNet2020Dataset.collate_fn,
        )

        batch = next(iter(dl))
        out = m(batch)

        self.assertEqual(out.shape, (2, 9))
