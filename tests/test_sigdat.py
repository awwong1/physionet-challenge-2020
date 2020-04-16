import os
import unittest
from tempfile import TemporaryDirectory

from wfdb.io import rdrecord

from driver import load_challenge_data
from util.sigdat import convert_to_wfdb_record, extract_features


class SigDatTest(unittest.TestCase):
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

        self.input_files = tuple(input_files)

    def test_parse_existing_dat(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dat_path = os.path.join(dir_path, "dat_data", "s0010_re.dat")
        with open(dat_path, "rb") as f:
            a = f.read()
        self.assertEqual(len(a), 921600)
        a_hex = a.hex()
        self.assertEqual(len(a_hex), 1843200)

    def test_read_wfdb_sample(self):
        tmp_input_file = os.path.join(self.input_directory, "A0001")
        r = rdrecord(tmp_input_file, physical=False)

        pass

    def test_write_wfdb_sample(self):
        tmp_input_file = os.path.join(self.input_directory, "A0001.mat")

        data, headers = load_challenge_data(tmp_input_file)

        with TemporaryDirectory() as temp_dir:
            r = convert_to_wfdb_record(data, headers)
            r.adc(inplace=True)
            r.wrsamp(write_dir=temp_dir)
            self.assertCountEqual(os.listdir(temp_dir), ["A0001.mat", "A0001.hea"])

        self.assertFalse(os.path.isdir(temp_dir), "temp dir not cleaned up")

    def test_extract_features(self):
        # tmp_input_file = os.path.join(self.input_directory, "A0001.mat")
        tmp_input_file = os.path.join(self.input_directory, "A0002.mat")
        # tmp_input_file = os.path.join(self.input_directory, "A0016.mat")
        # tmp_input_file = os.path.join(self.input_directory, "A1140.mat")
        # tmp_input_file = os.path.join(self.input_directory, "A3146.mat")
        data, headers = load_challenge_data(tmp_input_file)
        r = convert_to_wfdb_record(data, headers)

        features = extract_features(r)
        self.assertCountEqual(features.keys(), ["age", "sex", "sig", "target", "meta"])
        self.assertCountEqual(
            features["sig"].keys(),
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        )
        for k, v in features["sig"].items():
            self.assertCountEqual(
                v.keys(),
                [
                    "R-peak",
                    "P-peak",
                    "T-peak",
                    "HR",
                    "P-wave",
                    "R-wave",
                    "T-wave",
                    "TT-wave",
                    "PR-interval",
                    "PR-segment",
                    "ST-interval",
                    "ST-segment",
                    "Nt-wave",
                    "Nil-wave",
                    "pN-wave",
                    "Np-wave",
                    "NN-wave",
                    "FFT",
                    "polyfit_18",
                    "num_unsupported_symbols"
                ],
            )
