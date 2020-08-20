import json
import unittest
from glob import glob

import joblib
import pandas as pd
import tsfresh
import wfdb

from neurokit2_parallel import (
    ECG_LEAD_NAMES,
    ecg_clean,
    lead_to_feature_dataframe,
    parse_comments,
)
from util import parse_fc_parameters


class TestParseFCParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_mat_records = tuple(sorted(glob("tests/data/*.mat")))

        with open("importances_rank.json") as f:
            importances = json.load(f)

        cls.important_features = importances["sorted_keys"]

    def test_parse_fc_parameters(self):
        parsed_fcs = parse_fc_parameters(self.important_features)

        reference_fc = tsfresh.feature_extraction.ComprehensiveFCParameters()

        for lead_feat_type, parsed_fc in parsed_fcs.items():
            if parsed_fc is None:
                continue
            for feat_key, parsed_feat_val in parsed_fc.items():
                ref_feat_val = reference_fc.get(feat_key)

                # shim the 'augmented_dickey_fuller' for test
                if feat_key == "augmented_dickey_fuller":
                    ref_feat_val = [{**kv, "autolag": "AIC"} for kv in ref_feat_val]

                if type(parsed_feat_val) is list:
                    self.assertCountEqual(
                        parsed_feat_val, ref_feat_val, f"check {feat_key}"
                    )
                else:
                    self.assertEqual(parsed_feat_val, ref_feat_val, f"check {feat_key}")

    def test_feature_extraction_with_importances(self):
        # LIMIT TO top 1000 features
        fc_parameters = parse_fc_parameters(self.important_features[:1000])

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
            # record_features = joblib.Parallel(n_jobs=num_leads, verbose=0)(
            #     joblib.delayed(lead_to_feature_dataframe)(
            #         r.p_signal[:, i], cleaned_signals[:, i], ECG_LEAD_NAMES[i], r.fs, fc_parameters
            #     )
            #     for i in range(num_leads)
            # )

            # single process version
            record_features = [
                lead_to_feature_dataframe(
                    r.p_signal[:, i],
                    cleaned_signals[:, i],
                    ECG_LEAD_NAMES[i],
                    r.fs,
                    fc_parameters,
                )
                for i in range(num_leads)
            ]

            # meta features
            meta_dict = {}
            if fc_parameters:
                if "age" in fc_parameters:
                    meta_dict["age"] = (age,)
                if "sex" in fc_parameters:
                    meta_dict["sex"] = (sex,)
            else:
                meta_dict = {"age": (age,), "sex": (sex,)}

            record_features = pd.concat(
                [pd.DataFrame(meta_dict)] + record_features, axis=1
            )

            self.assertCountEqual(
                record_features.columns.to_list(), self.important_features[:1000]
            )
