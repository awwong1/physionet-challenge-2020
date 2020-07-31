import unittest
from glob import glob

import numpy as np
import pandas as pd
import wfdb

import neurokit2 as nk

from neurokit2_parallel import ecg_clean, ecg_peaks, signal_rate, ecg_quality


class TestNeurokit2Parallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure that glue code for converting to WFDB matches reference for all test records
        cls.all_mat_records = tuple(sorted(glob("tests/data/*.mat")))

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
        # for mat_record_fp in self.all_mat_records:
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            try:
                r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

                cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)
                sig_len, num_leads = cleaned_signals.shape

                output_peaks = []
                for lead_idx in range(num_leads):
                    try:
                        output_peaks.append(
                            nk.ecg_peaks(
                                cleaned_signals[:, lead_idx],
                                sampling_rate=r.fs,
                                correct_artifacts=True,
                            )
                        )
                    except Exception:
                        # Q2428 bad lead 2, 5
                        # parallelized code handles the edge case where no R-peaks detected
                        output_peaks.append(
                            (
                                pd.DataFrame({"ECG_R_Peaks": [0.0,] * sig_len}),
                                {"ECG_R_Peaks": np.array([])},
                            )
                        )
                        pass

                signals, info = ecg_peaks(
                    cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
                )
                for lead_idx, output_peak in enumerate(output_peaks):
                    ref_signal, ref_info = output_peak
                    par_signal = signals[
                        signals["ECG_Sig_Name"] == r.sig_name[lead_idx]
                    ]
                    par_info = info[lead_idx]

                    self.assertTrue(
                        (par_signal["ECG_R_Peaks"] == ref_signal["ECG_R_Peaks"]).all()
                    )
                    self.assertTrue(
                        (ref_info["ECG_R_Peaks"] == par_info["ECG_R_Peaks"]).all()
                    )
            except Exception:
                raise Exception(mat_record_fp)

    def test_signal_rate(self):
        # for mat_record_fp in self.all_mat_records:
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)
            sig_len, num_leads = cleaned_signals.shape

            signals, info = ecg_peaks(
                cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
            )

            ref_rates = {}
            for lead_idx, sig_name in enumerate(r.sig_name):
                # par_signal = signals[
                #     signals["ECG_Sig_Name"] == r.sig_name[lead_idx]
                # ]
                par_info = info[lead_idx]

                ref_rate = nk.signal_rate(
                    par_info, sampling_rate=r.fs, desired_length=sig_len
                )
                ref_rates[lead_idx] = ref_rate

            par_rate = signal_rate(info, sampling_rate=r.fs, desired_length=sig_len)
            for k, ref_rate in ref_rates.items():
                if not (par_rate[k] == ref_rate).all():
                    # check that they are both all NaN
                    self.assertTrue(
                        np.isnan(ref_rate).all() and np.isnan(par_rate[k]).all(),
                        f"{mat_record_fp} lead_idx {lead_idx}",
                    )
                else:
                    self.assertTrue(
                        (par_rate[k] == ref_rate).all(),
                        f"{mat_record_fp} lead_idx {lead_idx}",
                    )

    def test_ecg_quality(self):
        # for mat_record_fp in self.all_mat_records:
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)
            sig_len, num_leads = cleaned_signals.shape

            signals, info = ecg_peaks(
                cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
            )

            ref_quality = {}
            for lead_idx, sig_name in enumerate(r.sig_name):
                ecg_cleaned = cleaned_signals[:, lead_idx]
                rpeaks = info[lead_idx]["ECG_R_Peaks"]

                ref = nk.ecg_quality(
                    ecg_cleaned, rpeaks=rpeaks, sampling_rate=r.fs
                )
                ref_quality[lead_idx] = ref

            ecg_quality(cleaned_signals, info, sampling_rate=r.fs)

            self.assertTrue(False, "todo")
