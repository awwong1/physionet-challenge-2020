# -*- coding: utf-8 -*-
import json
import math
import os
import re
from glob import glob
from multiprocessing import Pool, freeze_support

import torch
import wfdb
from scipy import signal
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ["PhysioNet2020Dataset"]


class PhysioNet2020Dataset(Dataset):
    """PhysioNet 2020 Challenge Dataset.
    """

    BASE_FS = 500
    LABELS = ("Normal", "AF", "I-AVB", "LBBB",
              "RBBB", "PAC", "PVC", "STD", "STE")

    def __init__(
        self, data_dir, fs=BASE_FS, max_seq_len=2000, proc=0, ensure_equal_len=False, signal_last=False
    ):
        """Initialize the PhysioNet 2020 Challenge Dataset
        data_dir: path to *.hea files
        fs: Sampling frequency to return tensors as (default: 500 samples per second)
        max_seq_len: Maximum length of returned tensors (default: 2000 samples per tensor)
        proc: Number of processes for generating dataset length references (0: single threaded)
        """
        super(PhysioNet2020Dataset, self).__init__()

        self.data_dir = data_dir
        self.fs = fs
        self.max_seq_len = max_seq_len
        self.proc = proc
        self.ensure_equal_len = ensure_equal_len
        self.signal_last = signal_last  # if true, yield signals as (CHANNEL, SIGNAL)
        self.record_lens = {}  # length of each record (sample size)
        self.record_samps = {}  # number of max_seq_len segments per record
        self.idx_to_key_rel = {}  # dataset index to record key

        self.record_names = sorted(
            [fn.split(".hea")[0]
             for fn in glob(os.path.join(data_dir, "*.hea"))]
        )
        self.initialize_length_references()

    def __len__(self):
        return sum(self.record_samps.values())

    def initialize_length_references(self):
        # have the dataset length references been constructed?
        ds_len_ref_pth = os.path.join(self.data_dir, ".len_ref_500hz.json")
        try:
            with open(ds_len_ref_pth, "r") as f:
                self.record_lens = json.load(f)
        except Exception:
            # if not, build and save the dataset length references
            with tqdm(
                self.record_names, desc="Building dataset length references"
            ) as t:
                if self.proc <= 0:
                    self.record_lens = dict(
                        PhysioNet2020Dataset.derive_meta(rn) for rn in t
                    )
                else:
                    freeze_support()
                    with Pool(
                        self.proc,
                        initializer=tqdm.set_lock,
                        initargs=(tqdm.get_lock(),),
                    ) as p:
                        self.record_lens = dict(
                            p.imap_unordered(
                                PhysioNet2020Dataset._derive_meta, t)
                        )
            with open(ds_len_ref_pth, "w") as f:
                json.dump(self.record_lens, f)

        # store the lengths and sample counts for each of the records
        for rn in self.record_names:
            len_500hz = self.record_lens[rn]
            # resample to specified fs
            len_resamp = int(
                len_500hz / PhysioNet2020Dataset.BASE_FS * self.fs)
            self.record_lens[rn] = len_resamp

            r_num_samples = math.ceil(len_resamp / self.max_seq_len)
            r_start_idx = sum(self.record_samps.values())
            r_end_idx = r_start_idx + max(r_num_samples, 0)
            for idx in range(r_start_idx, r_end_idx):
                self.idx_to_key_rel[idx] = (rn, idx - r_start_idx)
            self.record_samps[rn] = r_num_samples

            continue

    def __getitem__(self, idx):
        r_pth, rel_idx = self.idx_to_key_rel[idx]
        r = wfdb.rdrecord(r_pth)
        _raw_age, _raw_sx, raw_dx, _rx, _hx, _sx = r.comments
        dx_grp = re.search(r"^Dx: (?P<dx>.*)$", raw_dx)

        # resample the signal to new frequency
        # sig, _ = processing.resample_sig(r.p_signal, self.BASE_FS, self.fs)
        sig = signal.resample(r.p_signal, self.record_lens[r_pth])
        # offset by the relative index
        start_idx = self.max_seq_len * rel_idx
        end_idx = min(len(sig), start_idx + self.max_seq_len)
        if self.ensure_equal_len:
            # start to end must equal max_seq_len
            sig = torch.FloatTensor(sig[end_idx - self.max_seq_len: end_idx])
            if len(sig) < self.max_seq_len:
                pad = self.max_seq_len - len(sig)
                sig = torch.nn.functional.pad(
                    sig, (0, 0, pad, 0), "constant", 0)
        else:
            sig = torch.FloatTensor(sig[start_idx:end_idx])

        if self.signal_last:
            sig = sig.transpose(0, 1)

        target = [0.0] * len(self.LABELS)
        for dxi in dx_grp.group("dx").split(","):
            target[self.LABELS.index(dxi)] = 1
        target = torch.FloatTensor(target)

        return sig, target

    @staticmethod
    def _derive_meta(rn):
        r = wfdb.rdrecord(rn)
        return rn, r.sig_len

    @staticmethod
    def collate_fn(batch, pad_value=0):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=pad_value)

        return xx_pad, yy_pad, x_lens, y_lens
