import os
import unittest

import joblib
import numpy as np

from util.evaluation_helper import evaluate_score_batch
from feature_extractor import structured_np_array_to_features


class TestEvaluationHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Find files.
        input_files = []
        for f in os.listdir("tests/data"):
            if (
                os.path.isfile(os.path.join("tests/data", f))
                and not f.lower().startswith(".")
                and f.lower().endswith("mat")
            ):
                input_files.append(f)

        cls.input_files = input_files

    @unittest.skip("TODO: too slow with large .data_cache.sav")
    def test_evaluation_helper(self):
        # load the trained test model
        loaded_model = joblib.load("tests/model/finalized_model_1594757805.sav")
        data_cache = joblib.load(".data_cache.sav", mmap_mode="r")

        test_files = sorted(
            [
                "data/WFDB/E00009",
                "data/Training_2/Q0001",
                "data/Training_2/Q0027",
                "data/Training_2/Q0122",
                "data/Training_2/Q1808",
                "data/WFDB/E00015",
                "data/Training_2/Q0010",
                "data/Training_2/Q0033",
                "data/Training_2/Q0398",
                "data/Training_2/Q1847",
                "data/WFDB/E00018",
                "data/Training_2/Q0012",
                "data/Training_2/Q0048",
                "data/Training_2/Q0630",
                "data/Training_2/Q1917",
                "data/WFDB/E00793",
                "data/Training_2/Q0017",
                "data/Training_2/Q0050",
                "data/Training_2/Q1240",
                "data/Training_2/Q2428",
                "data/WFDB/HR00599",
                "data/Training_2/Q0022",
                "data/Training_2/Q0077",
                "data/Training_2/Q1804",
                "data/WFDB/HR20715",
                "data/Training_2/Q0026",
                "data/Training_2/Q0096",
                "data/Training_2/Q1805",
            ]
        )

        classes = []
        labels = []
        scores = []
        ground_truth = []

        for tf_idx, test_file in enumerate(test_files):
            struct_np_arr = data_cache[data_cache["record_name"] == test_file]
            features = structured_np_array_to_features(struct_np_arr)

            file_labels = []
            file_scores = []
            for k, v in loaded_model.items():
                if k in ("train_records", "eval_records"):
                    continue
                if tf_idx == 0:
                    classes.append(str(k))

                file_labels.append(int(v.predict(features)[0]))
                file_scores.append(v.predict_proba(features)[0][1])

            labels.append(file_labels)
            scores.append(file_scores)
            ground_truth.append([str(dx) for dx in struct_np_arr["dx"][0]])

        labels = np.array(labels)
        scores = np.array(scores)

        evaluation = evaluate_score_batch(
            predicted_classes=classes,
            predicted_labels=labels,
            predicted_probabilities=scores,
            raw_ground_truth_labels=ground_truth,
        )
        (
            auroc,
            auprc,
            accuracy,
            f_measure,
            f_beta_measure,
            g_beta_measure,
            challenge_metric,
        ) = evaluation

        self.assertEqual(f"{auroc:.3f}", "0.918")
        self.assertEqual(f"{auprc:.3f}", "0.741")
        self.assertEqual(f"{accuracy:.3f}", "0.393")
        self.assertEqual(f"{f_measure:.3f}", "0.647")
        self.assertEqual(f"{f_beta_measure:.3f}", "0.699")
        self.assertEqual(f"{g_beta_measure:.3f}", "0.568")
        self.assertEqual(f"{challenge_metric:.3f}", "0.691")
