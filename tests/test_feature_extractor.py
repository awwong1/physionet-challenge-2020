import unittest

import numpy as np
import wfdb

from feature_extractor import (
    get_structured_lead_features,
    IR_COLS,
    HB_SIG_TSFRESH_COLS,
    LEAD_SIG_TSFRESH_COLS,
    hea_fp_to_np_array
)


@unittest.skip("Feature Extractor class no longer used")
class TestFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_cols = IR_COLS + HB_SIG_TSFRESH_COLS + LEAD_SIG_TSFRESH_COLS

    def test_get_structured_lead_features(self):
        r = wfdb.rdrecord("tests/data/Q0001")

        sig_len, num_leads = r.p_signal.shape

        for lead_idx in range(num_leads):
            lead_sig = r.p_signal[:, lead_idx]
            lead_name = r.sig_name[lead_idx]
            data, dtype = get_structured_lead_features(
                lead_sig, sampling_rate=r.fs, lead_name=lead_name,
            )

            self.assertEqual(len(data), len(self.data_cols))

            self.assertCountEqual(
                [d[0] for d in dtype],
                [f"{lead_name}_{d_col}" for d_col in self.data_cols],
            )

    def test_get_structured_lead_features_on_bad_data(self):
        lead_sig = np.zeros(5000)
        lead_name = "ZRO"
        data, dtype = get_structured_lead_features(
            lead_sig, sampling_rate=500, lead_name=lead_name,
        )

        self.assertEqual(len(data), len(self.data_cols))

        self.assertCountEqual(
            [d[0] for d in dtype], [f"{lead_name}_{d_col}" for d_col in self.data_cols]
        )

        self.assertTrue(any(np.isnan(data)))

        lead_sig = np.random.rand(5000)
        lead_name = "RND"
        data, dtype = get_structured_lead_features(
            lead_sig, sampling_rate=500, lead_name=lead_name,
        )

        self.assertEqual(len(data), len(self.data_cols))

        self.assertCountEqual(
            [d[0] for d in dtype], [f"{lead_name}_{d_col}" for d_col in self.data_cols]
        )

        self.assertTrue(any(np.isnan(data)))
        # self.assertTrue(any(np.isfinite(data)))

    def test_hea_fp_to_np_array(self):
        str_arr = hea_fp_to_np_array("tests/data/Q0001.hea")

        ref_dtype = [
            ("record_name", "<U50"),
            ("seq_len", "<f8"),
            ("sampling_rate", "<f8"),
            ("age", "<f8"),
            ("sex", "<f8"),
            ("dx", "|O"),
        ]
        lead_names = (
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        )

        for lead_name in lead_names:
            ref_dtype += [(f"{lead_name}_{d_col}", "<f8") for d_col in self.data_cols]

        self.assertEqual(str_arr.shape, (1,))
        self.assertEqual(str_arr.dtype.descr, ref_dtype)
