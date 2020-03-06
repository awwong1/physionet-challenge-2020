import os
import unittest

from torch.utils.data import DataLoader
from datasets import PhysioNet2020Dataset


class PhysioNet2020DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        try:
            self.proc = len(os.sched_getaffinity(0))
        except:
            self.proc = 0

    def test_base_ds(self):
        ds = PhysioNet2020Dataset("Training_WFDB")

        self.assertEqual(len(ds), 6877)
        self.assertEqual(max(ds.idx_to_key_rel.keys()), 6876)

        dl = DataLoader(
            ds,
            batch_size=23,
            num_workers=self.proc,
            collate_fn=PhysioNet2020Dataset.collate_fn,
        )
        for idx, batch in enumerate(dl):
            self.assertEqual(
                list(batch.keys()), ["signal", "target", "sex", "age", "len"]
            )
            self.assertEqual(batch["target"].shape, (23, 9))
            self.assertGreaterEqual(batch["signal"].size(0), 100)
            self.assertEqual(batch["signal"].size(1), 23)
            self.assertEqual(batch["signal"].size(2), 12)
        self.assertEqual(idx, 298)

    def test_ds_max_seq_len_eq_len(self):
        ds = PhysioNet2020Dataset(
            "Training_WFDB", max_seq_len=4000, ensure_equal_len=True
        )

        self.assertEqual(len(ds), 17541)
        self.assertEqual(max(ds.idx_to_key_rel.keys()), 17540)

        dl = DataLoader(
            ds,
            batch_size=9,
            num_workers=self.proc,
            collate_fn=PhysioNet2020Dataset.collate_fn,
        )
        for idx, batch in enumerate(dl):
            self.assertEqual(
                list(batch.keys()), ["signal", "target", "sex", "age", "len"]
            )
            self.assertEqual(batch["target"].shape, (9, 9))
            self.assertEqual(batch["signal"].shape, (4000, 9, 12))
        self.assertEqual(idx, 1948)

    def test_split_names_cv(self):
        train_records, val_records = PhysioNet2020Dataset.split_names_cv(
            "Training_WFDB", 5, 0
        )
        self.assertEqual(len(train_records), 5501)
        self.assertEqual(len(val_records), 1376)
        self.assertEqual(train_records[0], "A1377")
        self.assertEqual(train_records[-1], "A6877")
        self.assertEqual(val_records[0], "A0001")
        self.assertEqual(val_records[-1], "A1376")

        train_records, val_records = PhysioNet2020Dataset.split_names_cv(
            "Training_WFDB", 5, 1
        )
        self.assertEqual(len(train_records), 5501)
        self.assertEqual(len(val_records), 1376)
        self.assertEqual(train_records[0], "A0001")
        self.assertEqual(train_records[-1], "A6877")
        self.assertEqual(val_records[0], "A1377")
        self.assertEqual(val_records[-1], "A2752")

        train_records, val_records = PhysioNet2020Dataset.split_names_cv(
            "Training_WFDB", 5, 4
        )
        self.assertEqual(len(train_records), 5504)
        self.assertEqual(len(val_records), 1373)
        self.assertEqual(train_records[0], "A0001")
        self.assertEqual(train_records[-1], "A5504")
        self.assertEqual(val_records[0], "A5505")
        self.assertEqual(val_records[-1], "A6877")
