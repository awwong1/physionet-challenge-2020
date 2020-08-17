import unittest
from glob import glob

import numpy as np
import pandas as pd
import joblib
import tsfresh
import wfdb

import neurokit2 as nk

from neurokit2_parallel import (
    ECG_LEAD_NAMES,
    KEYS_INTERVALRELATED,
    KEYS_TSFRESH,
    FC_PARAMETERS,
    ecg_clean,
    ecg_peaks,
    signal_rate,
    ecg_quality,
    ecg_delineate,
    ecg_intervalrelated,
    get_intervalrelated_features,
    best_heartbeats_from_ecg_signal,
    signal_to_tsfresh_df,
    lead_to_feature_dataframe,
    parse_comments,
)


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

                try:
                    ref = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks, sampling_rate=r.fs)
                except Exception:
                    ref = np.full(sig_len, np.nan)
                ref_quality[lead_idx] = ref

            par_quality = ecg_quality(cleaned_signals, info, sampling_rate=r.fs)

            for k, v in ref_quality.items():
                if not (par_quality[k] == v).all():
                    # check that they are both all NaN
                    self.assertTrue(
                        np.isnan(v).all() and np.isnan(par_quality[k]).all(),
                        f"{mat_record_fp} lead_idx {lead_idx}",
                    )
                else:
                    self.assertTrue(
                        (par_quality[k] == v).all(),
                        f"{mat_record_fp} lead_idx {lead_idx}",
                    )

    def test_ecg_delineate(self):
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

            ref_delineate = []
            for lead_idx, sig_name in enumerate(r.sig_name):
                ecg_cleaned = cleaned_signals[:, lead_idx]
                rpeaks = info[lead_idx]

                try:
                    ref = nk.ecg_delineate(
                        ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=r.fs
                    )
                except Exception:
                    ref = (
                        pd.DataFrame(
                            {
                                "ECG_P_Peaks": [0.0,] * sig_len,
                                "ECG_Q_Peaks": [0.0,] * sig_len,
                                "ECG_S_Peaks": [0.0,] * sig_len,
                                "ECG_T_Peaks": [0.0,] * sig_len,
                                "ECG_P_Onsets": [0.0,] * sig_len,
                                "ECG_T_Offsets": [0.0,] * sig_len,
                            }
                        ),
                        {
                            "ECG_P_Peaks": [],
                            "ECG_Q_Peaks": [],
                            "ECG_S_Peaks": [],
                            "ECG_T_Peaks": [],
                            "ECG_P_Onsets": [],
                            "ECG_T_Offsets": [],
                        },
                    )
                ref_delineate.append(ref)

            par_delineate_df, par_delineate_info = ecg_delineate(
                cleaned_signals, info, sampling_rate=r.fs, ecg_lead_names=r.sig_name
            )

            for lead_idx, output_ref in enumerate(ref_delineate):
                ref_delineate, ref_info = output_ref
                par_delineate_df_inst = par_delineate_df[
                    par_delineate_df["ECG_Sig_Name"] == r.sig_name[lead_idx]
                ]
                par_delineate_info_inst = par_delineate_info[lead_idx]

                self.assertTrue(ref_info == par_delineate_info_inst)

                for key in [
                    "ECG_P_Peaks",
                    "ECG_Q_Peaks",
                    "ECG_S_Peaks",
                    "ECG_T_Peaks",
                    "ECG_P_Onsets",
                    "ECG_T_Offsets",
                ]:
                    self.assertTrue(
                        (par_delineate_df_inst[key] == ref_delineate[key]).all()
                    )

    def test_ecg_intervalrelated(self):
        # for mat_record_fp in self.all_mat_records:
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)
            sig_len, num_leads = cleaned_signals.shape

            df_sig_names = []
            for ln in ECG_LEAD_NAMES:
                df_sig_names += [ln,] * sig_len

            peaks_df, peaks_info = ecg_peaks(
                cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
            )

            # non-df outputs...
            rate = signal_rate(peaks_info, sampling_rate=r.fs, desired_length=sig_len)
            quality = ecg_quality(cleaned_signals, peaks_info, sampling_rate=r.fs)

            rate_values = np.concatenate(
                [rate[lead_idx] for lead_idx in range(num_leads)]
            )
            quality_values = np.concatenate(
                [quality[lead_idx] for lead_idx in range(num_leads)]
            )

            proc_df = pd.DataFrame(
                {
                    "ECG_Raw": r.p_signal.flatten(order="F"),
                    "ECG_Clean": cleaned_signals.flatten(order="F"),
                    "ECG_Sig_Name": df_sig_names,
                    "ECG_R_Peaks": peaks_df["ECG_R_Peaks"],
                    "ECG_Rate": rate_values,
                    "ECG_Quality": quality_values,
                }
            )

            record_feats = []
            for lead_name in ECG_LEAD_NAMES:
                try:
                    ip_df = proc_df[proc_df["ECG_Sig_Name"] == lead_name]
                    lead_feats = nk.ecg_intervalrelated(ip_df, sampling_rate=r.fs)
                except Exception:
                    lead_feats = pd.DataFrame.from_dict(
                        dict((k, (np.nan,)) for k in KEYS_INTERVALRELATED)
                    )
                finally:
                    lead_feats["ECG_Sig_Name"] = lead_name
                record_feats.append(lead_feats)

            record_feats = pd.concat(record_feats)
            whole_feats = ecg_intervalrelated(proc_df, sampling_rate=r.fs)

            self.assertTrue(
                ((record_feats.fillna(0.0) == whole_feats.fillna(0.0)).all()).all()
            )

    def test_get_intervalrelated_features(self):
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])
            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)

            ir_features, _ = get_intervalrelated_features(
                r.p_signal,
                cleaned_signals,
                sampling_rate=r.fs,
                ecg_lead_names=r.sig_name,
            )
            self.assertTrue(
                (ir_features.columns == KEYS_INTERVALRELATED + ["ECG_Sig_Name",]).all()
            )

    def test_best_heartbeats_from_ecg_signal(self):
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])
            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)

            _, proc_df = get_intervalrelated_features(
                r.p_signal,
                cleaned_signals,
                sampling_rate=r.fs,
                ecg_lead_names=r.sig_name,
            )

            lead_heartbeats = best_heartbeats_from_ecg_signal(
                proc_df, sampling_rate=r.fs, ecg_lead_names=r.sig_name
            )
            self.assertTrue(
                (lead_heartbeats.columns == ["lead", "time", "hb_sig"]).all()
            )

            hb_feats = tsfresh.extract_features(
                lead_heartbeats,
                column_id="lead",
                column_sort="time",
                disable_progressbar=True,
                default_fc_parameters=FC_PARAMETERS,
                n_jobs=0,
            )

            self.assertTrue(
                [k.split("hb_sig__")[1] for k in hb_feats.columns] == KEYS_TSFRESH
            )

    def test_signal_to_tsfresh_df(self):
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])
            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)

            lead_signals = signal_to_tsfresh_df(
                cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
            )
            self.assertTrue((lead_signals.columns == ["lead", "time", "sig"]).all())

            hb_feats = tsfresh.extract_features(
                lead_signals,
                column_id="lead",
                column_sort="time",
                disable_progressbar=True,
                default_fc_parameters=FC_PARAMETERS,
                n_jobs=0,
            )

            self.assertTrue(
                [k.split("sig__")[1] for k in hb_feats.columns] == KEYS_TSFRESH
            )

    def test_lead_to_feature_dataframe(self):
        for mat_record_fp in [
            "tests/data/E00793.mat",
            "tests/data/Q2428.mat",  # test leads with no detected R-peaks
        ]:
            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            age, sex, dx = parse_comments(r)
            r.sig_name = ECG_LEAD_NAMES  # force consistent naming
            cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)

            signal_length, num_leads = cleaned_signals.shape

            # each lead should be processed separately and then combined back together
            record_features = joblib.Parallel(n_jobs=num_leads, verbose=0)(
                joblib.delayed(lead_to_feature_dataframe)(
                    r.p_signal[:, i], cleaned_signals[:, i], ECG_LEAD_NAMES[i], r.fs
                )
                for i in range(num_leads)
            )
            record_features = pd.concat([
                pd.DataFrame({"age": (age,), "sex": (sex,)})
            ] + record_features, axis=1)

            self.assertEqual(record_features.shape, (1, 18950))
