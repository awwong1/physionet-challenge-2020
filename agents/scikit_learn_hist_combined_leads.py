import json
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from datasets import PhysioNet2020Dataset
from util.config import init_class
from util.evaluation import compute_auc, compute_beta_score

from . import BaseAgent


class ScikitLearnAgent(BaseAgent):
    """Custom logic for HistGradientBoostingClassifier, custom NaN support
    """

    LEADS = (
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

    LABELS = ("AF", "I-AVB", "LBBB", "Normal", "RBBB", "PAC", "PVC", "STD", "STE")

    def __init__(self, config):
        super(ScikitLearnAgent, self).__init__(config)

        # Initialize cross validation datasets
        self.data_dir = config.get(
            "data_dir", "experiments/PhysioNet2020/FeatureExtraction/out"
        )
        self.output_dir = config["out_dir"]
        cross_validation = config.get("cross_validation", None)

        if cross_validation:
            train_records, val_records = PhysioNet2020Dataset.split_names_cv(
                self.data_dir, endswith=".npz", **cross_validation
            )

            self.cv_tag = "cv{fold}-{val_offset}".format(**cross_validation)
        else:
            train_records = sorted(
                [
                    f[:-4]
                    for f in os.listdir(self.data_dir)
                    if (
                        os.path.isfile(os.path.join(self.data_dir, f))
                        and not f.lower().startswith(".")
                        and f.lower().endswith(".npz")
                    )
                ]
            )
            val_records = []
            self.cv_tag = "no-cv"

        self.logger.info(
            "Training on %d records (%s, ..., %s)",
            len(train_records),
            train_records[0],
            train_records[-1],
        )
        if val_records:
            self.logger.info(
                "Validating on %d records (%s, ..., %s)",
                len(val_records),
                val_records[0],
                val_records[-1],
            )
        else:
            self.logger.info("No validation records")

        self.train_records = train_records
        self.val_records = val_records

        self.use_multioutput = config.get("use_multioutput", False)
        self.classifier_chain_order = config.get("classifier_chain_order", None)
        self.convert_nans = config.get("convert_nans", False)

        self.classifiers = {
            label: HistGradientBoostingClassifier() for label in ScikitLearnAgent.LABELS
        }
        self.classifier_name = "HistGradientBoostingClassifier"

        self.logger.info(self.classifiers["Normal"])

    def run(self):
        inputs, targets = self.prepare_data(self.train_records, mode="Training")
        if self.use_multioutput:
            for label_idx, label in enumerate(ScikitLearnAgent.LABELS):
                self.logger.info(
                    f"Fit training data {inputs.shape} to classifier {label}..."
                )
                start = datetime.now()
                self.classifiers[label].fit(inputs, targets[:, label_idx])
                self.logger.info(f"Took {datetime.now() - start}")
        elif self.classifier_chain_order:
            chain_inputs = inputs
            for label_idx in self.classifier_chain_order:
                label = ScikitLearnAgent.LABELS[label_idx]
                self.logger.info(
                    f"Fit training data {chain_inputs.shape} to classifier {label}..."
                )
                start = datetime.now()
                self.classifiers[label].fit(chain_inputs, targets[:, label_idx])
                self.logger.info(f"Took {datetime.now() - start}")
                chain_inputs = np.concatenate([chain_inputs, self.classifiers[label].predict_proba(chain_inputs)], axis=1)

        self.logger.info("Scoring training data...")
        start = datetime.now()
        self.evaluate_and_log(inputs, targets)
        self.logger.info(f"Took {datetime.now() - start}")

        if self.val_records:
            inputs, targets = self.prepare_data(self.val_records, mode="Validation")
            self.logger.info("Scoring validation data...")
            start = datetime.now()
            self.evaluate_and_log(inputs, targets, mode="Validation")
            self.logger.info(f"Took {datetime.now() - start}")

    def prepare_data(self, records, mode="Training"):
        self.logger.info(f"Preparing {mode} data...")
        start = datetime.now()
        raw_data = map(self.load_data_cache, records)
        inputs, targets = zip(*raw_data)

        stacked_inputs = np.stack(inputs)
        stacked_targets = np.stack(targets)
        self.logger.info(f"Took {datetime.now() - start}")
        return stacked_inputs, stacked_targets

    def load_data_cache(self, train_record):
        fp = os.path.join(self.data_dir, f"{train_record}.npz")
        data = np.load(fp)

        # raw_data = np.concatenate([data[k] for k in ScikitLearnAgent.LEADS])
        # only record sex (2) and age (1) once
        raw_data = np.concatenate(
            [data["I"][:3]] + [data[k][3:] for k in ScikitLearnAgent.LEADS]
        )

        if self.convert_nans:
            raw_data = np.where(raw_data == -1000, float("nan"), raw_data)

        return (
            raw_data,
            data["target"],
        )

    def evaluate_and_log(self, inputs, targets, mode="Training"):
        # evaluate each individual lead classifier
        probabilities = {}
        outputs = {}
        if self.use_multioutput:
            for label_idx, label in enumerate(ScikitLearnAgent.LABELS):
                probabilities[label] = self.classifiers[label].predict_proba(inputs)
                outputs[label] = self.classifiers[label].predict(inputs)
        elif self.classifier_chain_order:
            chain_inputs = inputs
            for label_idx in self.classifier_chain_order:
                label = ScikitLearnAgent.LABELS[label_idx]
                probabilities[label] = self.classifiers[label].predict_proba(chain_inputs)
                outputs[label] = self.classifiers[label].predict(chain_inputs)
                chain_inputs = np.concatenate([chain_inputs, probabilities[label]], axis=1)

        probabilities = np.stack([probabilities[label][:, 1] for label in self.LABELS]).T
        outputs = np.stack([outputs[label] for label in self.LABELS]).T
        targets = np.stack(targets)

        # accuracy, f_measure, f_beta, g_beta
        beta_score = compute_beta_score(targets, outputs)
        # auroc, auprc
        auc_score = compute_auc(targets, probabilities)
        scores = np.array(beta_score + auc_score)
        self.logger.info(f"{self.cv_tag}/{mode} mean scores:")

        pipeline = ""
        if self.use_multioutput:
            pipeline += "CustomMultiOutput"
        elif self.classifier_chain_order:
            pipeline += f"CustomChain(order={self.classifier_chain_order})"
        if self.convert_nans:
            pipeline += "-NaN"

        header_line = "| Classifier | Pipeline | Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |"
        hbreak_line = "|------------|----------|---------|----------|-----------|--------|--------|-------|-------|"

        output_line = (
            f"| {self.classifier_name} | {pipeline} | "
            + f"{self.cv_tag}/{mode} | "
            + f"{scores[0]:.4f} | {scores[1]:.4f} | "
            + f"{scores[2]:.4f} | {scores[3]:.4f} | "
            + f"{scores[4]:.4f} | {scores[5]:.4f} |"
        )

        self.logger.info(header_line)
        self.logger.info(hbreak_line)
        self.logger.info(output_line)

        wp = os.path.join(self.output_dir, "scores.txt")
        file_exists = os.path.isfile(wp)
        with open(wp, "a") as f:
            if not file_exists:
                f.write(header_line + "\n")
                f.write(hbreak_line + "\n")
            f.write(output_line + "\n")
