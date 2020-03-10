import os
import unittest

from torch.utils.data import DataLoader
from datasets import PhysioNet2020Dataset
from models.sig_fft_lstm import SignalFourierTransformLSTM


class SignalFourierTransformLSTMTest(unittest.TestCase):
    def test_forward(self):
        ds = PhysioNet2020Dataset(
            "Training_WFDB",
            max_seq_len=4000,
            records=("A0001", "A0002", "A0003", "A0004", "A0005", "A0006", "A0007"),
            proc=0,
            derive_fft=True,
        )
        m = SignalFourierTransformLSTM()
        dl = DataLoader(
            ds,
            batch_size=8,
            num_workers=0,
            collate_fn=PhysioNet2020Dataset.collate_fn,
            shuffle=True,
            pin_memory=True,
        )

        batch = next(iter(dl))
        self.assertEqual(batch["signal"].shape, (4000, 8, 12))
        out = m(batch)

        self.assertEqual(out.shape, (8, 9))
