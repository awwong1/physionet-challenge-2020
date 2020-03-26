import os
import unittest
from driver import load_challenge_data
from util.sigdat import write_wfdb_sample
from wfdb.io import rdrecord

class DatConversionTest(unittest.TestCase):
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
        write_wfdb_sample(data, headers)
        pass
