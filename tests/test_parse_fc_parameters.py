import unittest
import json

import tsfresh

from util import parse_fc_parameters


class TestParseFCParameters(unittest.TestCase):
    def test_parse_fc_parameters(self):
        with open("importances_rank.json") as f:
            importances = json.load(f)

        important_features = importances["sorted_keys"]
        parsed_fcs = parse_fc_parameters(important_features)

        reference_fc = tsfresh.feature_extraction.ComprehensiveFCParameters()

        for lead_feat_type, parsed_fc in parsed_fcs.items():
            for feat_key, parsed_feat_val in parsed_fc.items():
                ref_feat_val = reference_fc.get(feat_key)

                # shim the 'augmented_dickey_fuller' for test
                if feat_key == "augmented_dickey_fuller":
                    ref_feat_val = [{**kv, "autolag": "AIC"} for kv in ref_feat_val]

                if type(parsed_feat_val) is list:
                    self.assertCountEqual(parsed_feat_val, ref_feat_val, f"check {feat_key}")
                else:
                    self.assertEqual(parsed_feat_val, ref_feat_val, f"check {feat_key}")
