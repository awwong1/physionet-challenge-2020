import unittest
from glob import glob

import numpy as np
import wfdb

import neurokit2 as nk

from neurokit2_parallel import ecg_clean, ecg_peaks


class TestNeurokit2Parallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure that glue code for converting to WFDB matches reference for all test records
        cls.all_mat_records = glob("tests/data/*.mat")

    def test_ecg_clean(self):
        for mat_record_fp in self.all_mat_records:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            sig_len, num_leads = r.p_signal.shape
            cleaned_leads = []
            for lead_idx in range(num_leads):
                cleaned_leads.append(
                    nk.ecg_clean(r.p_signal[:, lead_idx], sampling_rate=r.fs)
                )

            ref = np.vstack(cleaned_leads).T
            par = ecg_clean(r.p_signal, sampling_rate=r.fs)

            self.assertTrue((ref == par).all())
            self.assertEqual(par.shape, (sig_len, num_leads))

    def test_ecg_peaks(self):
        for mat_record_fp in self.all_mat_records:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)
            sig_len, num_leads = cleaned_signals.shape

            output_peaks = []
            for lead_idx in range(num_leads):
                output_peaks.append(
                    nk.ecg_peaks(
                        cleaned_signals[:, lead_idx],
                        sampling_rate=r.fs,
                        correct_artifacts=True,
                    )
                )

            ecg_peaks(cleaned_signals, sampling_rate=r.fs)
            self.assertTrue(False, "todo, implement")
