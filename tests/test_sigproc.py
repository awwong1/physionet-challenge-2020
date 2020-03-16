import os
import unittest

from driver import get_classes, load_challenge_data
from util.sigproc import extract_ecg_features, parse_header_data


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
        special_files = (
            "A0001.mat",
            "A0115.mat",
            "A0718.mat",
            "A3762.mat",
            "A5550.mat",
        )
        for input_file in special_files:

            # with tqdm(sorted(self.input_files, reverse=True)) as t:
            #     for input_file in t:
            tmp_input_file = os.path.join(self.input_directory, input_file)
            data, _ = load_challenge_data(tmp_input_file)
            feat = extract_ecg_features(data)

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

    def test_parse_header_data(self):
        header_data = [
            "A0001 12 500 7500 05-Feb-2020 11:39:16\n",
            "A0001.mat 16+24 1000/mV 16 0 28 -1716 0 I\n",
            "A0001.mat 16+24 1000/mV 16 0 7 2029 0 II\n",
            "A0001.mat 16+24 1000/mV 16 0 -21 3745 0 III\n",
            "A0001.mat 16+24 1000/mV 16 0 -17 3680 0 aVR\n",
            "A0001.mat 16+24 1000/mV 16 0 24 -2664 0 aVL\n",
            "A0001.mat 16+24 1000/mV 16 0 -7 -1499 0 aVF\n",
            "A0001.mat 16+24 1000/mV 16 0 -290 390 0 V1\n",
            "A0001.mat 16+24 1000/mV 16 0 -204 157 0 V2\n",
            "A0001.mat 16+24 1000/mV 16 0 -96 -2555 0 V3\n",
            "A0001.mat 16+24 1000/mV 16 0 -112 49 0 V4\n",
            "A0001.mat 16+24 1000/mV 16 0 -596 -321 0 V5\n",
            "A0001.mat 16+24 1000/mV 16 0 -16 -3112 0 V6\n",
            "#Age: 74\n",
            "#Sex: Male\n",
            "#Dx: RBBB\n",
            "#Rx: Unknown\n",
            "#Hx: Unknown\n",
            "#Sx: Unknows\n",
        ]
        record = parse_header_data(header_data)
        self.assertCountEqual(record.keys(), ("record", "signal", "comment"))
        self.assertEqual(record["comment"]["age"], 74)
        self.assertEqual(record["comment"]["sex"], [1, 0])
        self.assertEqual(record["comment"]["target"], [0, 0, 0, 0, 0, 0, 1, 0, 0])
