import unittest

import torch

from dataset import PhysioNet2020Dataset


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.ds = PhysioNet2020Dataset("Training_WFDB")

    def test_len(self):
        self.assertEqual(len(self.ds), 30344)
        self.assertEqual(max(self.ds.idx_to_key_rel.keys()), 30343)

        ds_200hz = PhysioNet2020Dataset("Training_WFDB", fs=200)
        self.assertEqual(len(ds_200hz), 13563)
        self.assertEqual(max(ds_200hz.idx_to_key_rel.keys()), 13562)

        ds_200hz_4000samp = PhysioNet2020Dataset("Training_WFDB", fs=200, max_seq_len=4000)
        self.assertEqual(len(ds_200hz_4000samp), 8376)
        self.assertEqual(max(ds_200hz_4000samp.idx_to_key_rel.keys()), 8375)

    def test_idx_ref(self):
        self.assertEqual(len(self.ds.idx_to_key_rel), 30344)
        self.assertEqual(self.ds.idx_to_key_rel[0], ("Training_WFDB/A0001", 0))
        self.assertEqual(self.ds.idx_to_key_rel[15172], ("Training_WFDB/A3456", 2))

    @unittest.skip("time consuming to run")
    def test_iterate_through_dataset(self):
        ds_200hz_8000samp = PhysioNet2020Dataset("Training_WFDB", fs=200, max_seq_len=8000)
        self.assertEqual(len(ds_200hz_8000samp), 7135)
        dl = torch.utils.data.DataLoader(
            ds_200hz_8000samp,
            batch_size=1
        )
        for idx, (signal, target) in enumerate(dl):
            batch_size, sig_len, channels = signal.shape
            self.assertEqual(batch_size, 1)
            self.assertLessEqual(sig_len, 8000)
            self.assertEqual(channels, 12)
            self.assertEqual(target.shape, (1,9))
        self.assertEqual(idx, len(ds_200hz_8000samp) - 1)

    def test_quick_lstm(self):
        signal, target = self.ds[0]
        self.assertEqual(signal.shape, (2000,12))
        target_val, target_idx = target.topk(1)
        self.assertEqual(target_idx, 4)
        self.assertEqual(target_val, 1.0)

        lstm = torch.nn.LSTM(input_size=12, hidden_size=22)
        # input must be (seq_len, batch, input_size)
        inputs = signal.reshape(2000, 1, 12).float()
        output, (hn, cn) = lstm(inputs)
        self.assertEqual(output.shape, (2000, 1, 22))

    def test_lstm_with_dataset_padding(self):
        ds_200hz_8000samp = PhysioNet2020Dataset("Training_WFDB", fs=200, max_seq_len=8000)
        self.assertEqual(len(ds_200hz_8000samp), 7135)
        dl = torch.utils.data.DataLoader(
            ds_200hz_8000samp,
            batch_size=5,
            collate_fn = PhysioNet2020Dataset.collate_fn
        )

        lstm = torch.nn.LSTM(input_size=12, hidden_size=22)

        for x_padded, y_padded, x_lens, y_lens in dl:
            x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_padded, x_lens, batch_first=True, enforce_sorted=False)

            output_packed, (hn, cn) = lstm(x_packed)

            output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True)

            batch_size, sig_len, channels = output_padded.shape

            self.assertEqual(batch_size, 5)
            self.assertLessEqual(sig_len, 8000)
            self.assertEqual(channels, 22)
            self.assertEqual(y_padded.shape, (5,9))
            break
