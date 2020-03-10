# -*- coding: utf-8 -*-
import json
import math
import os
import random
import re
from multiprocessing import Pool

import numpy as np
import scipy
import torch
import wfdb
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from util.config import init_class

__all__ = ["PhysioNet2020Dataset"]


class PhysioNet2020Dataset(Dataset):
    """PhysioNet 2020 Challenge Dataset.
    """

    BASE_FS = 500
    LABELS = ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
    SEX = ("Male", "Female")

    def __init__(
        self,
        data_dir,
        fs=BASE_FS,
        max_seq_len=None,
        ensure_equal_len=False,
        proc=None,
        records=None,
        derive_fft=False,
    ):
        """Initialize the PhysioNet 2020 Challenge Dataset
        data_dir: path to *.mat and *.hea files
        fs: Sampling frequency to return tensors as (default: 500 samples per second)
        max_seq_len: Maximum length of returned tensors (default: full signal length)
        ensure_equal_len: if True, set all signal lengths equal to ensure_equal_len
        proc: Number of processes for generating dataset length references (0: single threaded)
        records: only use provided record names (default: use all available records)
        derive_fft: include "fft" in signal features dictionary
        """
        super(PhysioNet2020Dataset, self).__init__()

        if proc is None:
            try:
                proc = len(os.sched_getaffinity(0))
            except Exception:
                proc = 0

        self.data_dir = data_dir
        self.fs = fs
        self.max_seq_len = max_seq_len
        self.proc = proc
        self.ensure_equal_len = ensure_equal_len
        self.derive_fft = derive_fft

        if ensure_equal_len:
            assert (
                self.max_seq_len is not None
            ), "Cannot ensure equal lengths on unbounded sequences"

        self.record_len_fs = {}  # length and fs of each record (sample size)
        self.record_samps = {}  # number of max_seq_len segments per record
        self.idx_to_key_rel = {}  # dataset index to record key

        record_names = []
        for f in os.listdir(self.data_dir):
            if (
                os.path.isfile(os.path.join(self.data_dir, f))
                and not f.lower().startswith(".")
                and f.lower().endswith(".hea")
            ):
                r_name = f[:-4]  # trim off .hea
                if records is not None and r_name not in records:
                    continue
                else:
                    record_names.append(r_name)
        self.record_names = tuple(sorted(record_names))

        self.initialize_length_references()

    def __len__(self):
        return sum(self.record_samps.values())

    def initialize_length_references(self):
        # construt the dataset length references
        if self.proc <= 0:
            self.record_len_fs = dict(
                PhysioNet2020Dataset._get_sig_len_fs(os.path.join(self.data_dir, rn))
                for rn in self.record_names
            )
        else:
            with Pool(self.proc) as p:
                self.record_len_fs = dict(
                    p.imap_unordered(
                        PhysioNet2020Dataset._get_sig_len_fs,
                        [os.path.join(self.data_dir, rn) for rn in self.record_names],
                    )
                )

        # store the lengths and sample counts for each of the records
        for rn in self.record_names:
            len_rn, fs_rn = self.record_len_fs[rn]
            if self.fs != fs_rn:
                len_resamp = int(len_rn / fs_rn * self.fs)
                self.record_len_fs[rn] = (len_resamp, self.fs)

            if self.max_seq_len is None:
                r_num_samples = 1
            else:
                r_num_samples = math.ceil(self.record_len_fs[rn][0] / self.max_seq_len)
            r_start_idx = sum(self.record_samps.values())
            r_end_idx = r_start_idx + max(r_num_samples, 0)
            for idx in range(r_start_idx, r_end_idx):
                self.idx_to_key_rel[idx] = (rn, idx - r_start_idx)
            self.record_samps[rn] = r_num_samples

    def __getitem__(self, idx):
        rn, rel_idx = self.idx_to_key_rel[idx]
        r = wfdb.rdrecord(os.path.join(self.data_dir, rn))
        raw_age, raw_sx, raw_dx, _rx, _hx, _sx = r.comments

        # resample the signal if necessary
        sig = r.p_signal  # shape SIGNAL, CHANNEL (e.g. 7500, 12)
        if self.fs != r.fs:
            sig = signal.resample(sig, self.record_len_fs[r_pth][0])

        # offset by the relative index if necessary
        if self.max_seq_len is None:
            start_idx = 0
            end_idx = len(sig)
        else:
            start_idx = self.max_seq_len * rel_idx
            end_idx = min(len(sig), start_idx + self.max_seq_len)
            if (end_idx - start_idx) < self.max_seq_len:
                # rather than just truncating it, push the start_idx back
                start_idx = max(0, end_idx - self.max_seq_len)

        # force shape matches ensure_equal_len if necessary
        if self.ensure_equal_len:
            # start to end must equal max_seq_len
            sig = torch.FloatTensor(sig[max(0, end_idx - self.max_seq_len) : end_idx])
            if len(sig) < self.max_seq_len:
                pad = self.max_seq_len - len(sig)
                sig = torch.nn.functional.pad(sig, (0, 0, pad, 0), "constant", 0)
        else:
            sig = torch.FloatTensor(sig[start_idx:end_idx])

        dx_grp = re.search(r"^Dx: (?P<dx>.*)$", raw_dx)
        target = [0.0] * len(PhysioNet2020Dataset.LABELS)
        for dxi in dx_grp.group("dx").split(","):
            target[PhysioNet2020Dataset.LABELS.index(dxi)] = 1.0
        target = torch.FloatTensor(target)

        age_grp = re.search(r"^Age: (?P<age>.*)$", raw_age)
        age = float(age_grp.group("age"))
        if math.isnan(age):
            age = -1
        age = torch.ByteTensor((age,))

        sx_grp = re.search(r"^Sex: (?P<sx>.*)$", raw_sx)
        sex = [0.0, 0.0]
        sex[PhysioNet2020Dataset.SEX.index(sx_grp.group("sx"))] = 1.0
        sex = torch.ByteTensor(sex)

        data = {
            "signal": sig,
            "target": target,
            "sex": sex,
            "age": age,
            "len": (len(sig),),
        }

        if self.derive_fft:
            data["fft"] = PhysioNet2020Dataset.derive_fft_from_signal(sig, fs=r.fs)

        return data

    @staticmethod
    def derive_fft_from_signal(sig, fs=500, rescale_0_1=True):
        """Signal is PyTorch Tensor in shape (seq_len, 12). For each of the 12 leads, perform FFT.
        """
        seq_len, num_leads = sig.shape
        fft = np.empty(shape=(seq_len // 2, num_leads))
        for lead_idx in range(num_leads):
            lead_sig = sig.detach().cpu()[:, lead_idx].numpy()

            # Window the signal
            window_sig = scipy.signal.hann(len(lead_sig)) * lead_sig

            # Apply FFT on windowed signal
            x = scipy.fft(window_sig)
            x_mag = scipy.absolute(x)
            f = np.linspace(0, fs, len(x_mag))  # unused, but when plotted will show only 0.5 fs bins valid
            # Only half of the bins are usable, discard the other half
            # plt.figure(figsize=(13, 5))
            # plt.plot(f[:len(f)//2], X_mag[:len(X_mag)//2])
            # plt.xlabel("Frequency (Hz)")
            # plt.show()

            fft[:, lead_idx] = x_mag[: (len(x_mag) // 2)]

        if rescale_0_1:
            # transform all values to fit between [0,1]
            scaler = MinMaxScaler()
            fft = scaler.fit_transform(fft)

        return torch.FloatTensor(fft)

    @staticmethod
    def split_names(data_dir, train_ratio):
        """Split all of the record names up into bins based on ratios
        """
        record_names = [
            f[:-4]
            for f in os.listdir(data_dir)
            if (
                os.path.isfile(os.path.join(data_dir, f))
                and not f.lower().startswith(".")
                and f.lower().endswith(".hea")
            )
        ]

        total_records_len = len(record_names)
        train_records_len = int(total_records_len * train_ratio)
        val_records_len = total_records_len - train_records_len

        train_records = tuple(random.sample(record_names, train_records_len))
        val_records = tuple(t for t in record_names if t not in train_records)

        return train_records, val_records

    @staticmethod
    def split_names_cv(data_dir, fold=5, val_offset=0):
        """Split the record names up into n-fold cross validation tuples
        fold: (int) number of cross validation sets (>= 2)
        val_offset: (int) current validation offset (0 < fold)
        """
        assert fold >= 2, f"fold must be greater than or equal to 2, got {fold}"
        assert (
            val_offset >= 0 and val_offset < fold
        ), f"val_offset must be between [0, {fold}), got {val_offset}"
        record_names = sorted(
            [
                f[:-4]
                for f in os.listdir(data_dir)
                if (
                    os.path.isfile(os.path.join(data_dir, f))
                    and not f.lower().startswith(".")
                    and f.lower().endswith(".hea")
                )
            ]
        )

        total_records_len = len(record_names)
        val_records_len = math.ceil(total_records_len / fold)

        val_s_idx = val_offset * val_records_len
        val_e_idx = val_s_idx + val_records_len

        val_records = tuple(record_names[val_s_idx:val_e_idx])
        train_records = tuple(t for t in record_names if not t in val_records)

        return train_records, val_records

    @staticmethod
    def _get_sig_len_fs(rn):
        r = wfdb.rdrecord(rn)
        return r.record_name, (r.sig_len, r.fs)

    @staticmethod
    def collate_fn(batch, pad=0):
        age = torch.stack(tuple(e["age"] for e in batch))
        sex = torch.stack(tuple(e["sex"] for e in batch))
        target = torch.stack(tuple(e["target"] for e in batch))
        signals = tuple(e["signal"] for e in batch)
        signal_lens = tuple(len(e) for e in signals)
        sig = pad_sequence(signals, padding_value=pad)

        data = {
            "signal": sig,
            "target": target,
            "sex": sex,
            "age": age,
            "len": signal_lens,
        }

        if all("fft" in e for e in batch):
            ffts = tuple(e["fft"] for e in batch)
            data["fft"] = pad_sequence(ffts, padding_value=pad)

        return data
