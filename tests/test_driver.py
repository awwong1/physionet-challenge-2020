import os
import unittest

from driver import load_challenge_data, get_classes


class DriverTest(unittest.TestCase):
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

    def test_input_files(self):
        self.assertEqual(len(self.input_files), 6877)

    def test_get_classes(self):
        classes = get_classes(self.input_directory, self.input_files)
        self.assertEqual(
            classes,
            ["AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"],
        )

    def test_load_challenge_data(self):
        tmp_input_file = os.path.join(self.input_directory, "A0001.mat")
        data, header_data = load_challenge_data(tmp_input_file)

        self.assertEqual(data.shape, (12, 7500))
        self.assertEqual(
            header_data,
            [
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
            ],
        )
