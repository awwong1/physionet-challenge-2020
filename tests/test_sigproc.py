import os
import unittest

from driver import load_challenge_data, get_classes
from util.sigproc import extract_ecg_features


class SigProcTest(unittest.TestCase):
    def setUp(self):
        self.input_directory = "Training_WFDB"
        input_files = []
        for f in os.listdir(self.input_directory):
            if (
                os.path.isfile(os.path.join(self.input_directory, f))
                and not f.lower().startswith(".")
                and f.lower().endswith("mat")
            ):
                input_files.append(f)

        self.input_files = tuple(sorted(input_files))

    def test_input_files(self):
        self.assertEqual(len(self.input_files), 6877)
        self.assertEqual(self.input_files[0], "A0001.mat")

    def test_extract_ecg_features(self):
        tmp_input_file = os.path.join(self.input_directory, "A0001.mat")
        data, _ = load_challenge_data(tmp_input_file)

        all_features = extract_ecg_features(data)
        self.assertCountEqual(
            (
                "ts",
                "filtered",
                "heart_rate",
                "heart_rate_ts",
                "rpeak_det",
                "rpeaks",
                "templates",
                "templates_ts",
            ),
            tuple(all_features.keys()),
        )
        for key in (
            "filtered",
            "heart_rate",
            "heart_rate_ts",
            "rpeaks",
            "templates",
            "templates_ts",
        ):
            self.assertEqual(len(all_features[key]), 12)

    def test_ecg_features_lead_consistent(self):
        # Sanity check feature extraction consistency
        # bad_files = ("A0115.mat", "A0718.mat")
        # for input_file in bad_files:
        for input_file in self.input_files:
            tmp_input_file = os.path.join(self.input_directory, input_file)
            data, _ = load_challenge_data(tmp_input_file)

            try:
                feat = extract_ecg_features(data)
            except Exception as e:
                self.assertIsNone(e, f"{input_file}, {e}")

            self.assertTrue(
                all(f.shape == feat["filtered"][0].shape for f in feat["filtered"]),
                f"{input_file} filtered outlier",
            )
            self.assertTrue(
                all(
                    f.shape == feat["templates_ts"][0].shape
                    for f in feat["templates_ts"]
                ),
                f"{input_file} templates_ts outlier",
            )
            # heartrate shape should always be 1 less than rpeaks shape
            for idx in range(12):
                self.assertGreater(
                    feat["rpeaks"][idx].shape[0],
                    feat["heart_rate"][idx].shape[0],
                    f"{input_file} heart_rate outlier lead {idx}",
                )
