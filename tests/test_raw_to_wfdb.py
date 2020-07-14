import unittest
from glob import glob

import wfdb

from util.raw_to_wfdb import convert_to_wfdb_record
from driver import load_challenge_data


class TestRawToWFDB(unittest.TestCase):
    def test_convert_to_wfdb_record(self):
        # Ensure that glue code for converting to WFDB matches reference for all test records
        all_mat_records = glob("tests/data/*.mat")

        for mat_record_fp in all_mat_records:

            r = wfdb.rdrecord(mat_record_fp.rsplit(".mat")[0])

            data, header_data = load_challenge_data(mat_record_fp)
            r_shim = convert_to_wfdb_record(data, header_data)

            self.assertEqual(r, r_shim)
