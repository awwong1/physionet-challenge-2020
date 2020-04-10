import json
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain

from datasets import PhysioNet2020Dataset
from util.config import init_class
from util.evaluation import compute_auc, compute_beta_score

from . import BaseAgent


class ScikitLearnAgent(BaseAgent):
    """Agent for PhysioNet 2020 challenge classification experiments using Scikit Learn classifiers

    Each lead gets its own classifier, then aggregate all results into another classifier
    All classifiers must support multilabel
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

    def __init__(self, config):
        super(ScikitLearnAgent, self).__init__(config)

        # Initialize cross validation datasets
        self.data_dir = config.get(
            "data_dir", "experiments/PhysioNet2020/FeatureExtraction/out"
        )
        self.output_dir = config["out_dir"]
        cross_validation = config.get("cross_validation", None)

        self.serialize_model = config.get("serialize_model", False)

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

        # todo make this classifier configuration dependent?
        self.classifier = RandomForestClassifier()
        self.classifier_name = "RandomForestClassifier"
        self.logger.info(self.classifier)

    def run(self):
        inputs, targets = self.prepare_data(
            self.train_records, mode="Training"
        )
        self.logger.info("Fit training data to classifier...")
        start = datetime.now()
        self.classifier.fit(inputs, targets)
        self.logger.info(f"Took {datetime.now() - start}")

        self.logger.info("Scoring training data...")
        start = datetime.now()
        self.evaluate_and_log(inputs, targets)
        self.logger.info(f"Took {datetime.now() - start}")

        if self.val_records:
            inputs, targets = self.prepare_data(
                self.val_records, mode="Validation"
            )
            self.logger.info("Scoring validation data...")
            start = datetime.now()
            self.evaluate_and_log(inputs, targets, mode="Validation")
            self.logger.info(f"Took {datetime.now() - start}")

    def finalize(self):
        pass

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
        raw_data = np.concatenate([data[k] for k in ScikitLearnAgent.LEADS])

        # enforce no nans
        input_data = np.where(np.isnan(raw_data), -10, raw_data)
        return (
            input_data,
            data["target"],
        )

    def evaluate_and_log(self, inputs, targets, mode="Training"):
        # evaluate each individual lead classifier
        probabilities = self.classifier.predict_proba(inputs)
        outputs = self.classifier.predict(inputs)
        targets = np.stack(targets)

        # convert probabilities into positive label probability matrix
        probabilities = np.stack([prob[:,1] for prob in probabilities]).T

        # accuracy, f_measure, f_beta, g_beta
        beta_score = compute_beta_score(targets, outputs)
        # auroc, auprc
        auc_score = compute_auc(targets, probabilities)
        scores = np.array(beta_score + auc_score)
        self.logger.info(f"{self.cv_tag}/{mode} mean scores:")
        self.logger.info(
            f"acc: {scores[0]} | f_measure: {scores[1]} | "
            + f"f_beta: {scores[2]} | g_beta: {scores[3]} | "
            + f"auroc: {scores[4]} | auprc: {scores[5]} "
        )

        wp = os.path.join(self.output_dir, "scores.txt")
        file_exists = os.path.isfile(wp)
        with open(wp, "a") as f:
            if not file_exists:
                f.write(
                    "| Classifier | Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |\n"
                    + "|------------|---------|----------|-----------|--------|--------|-------|-------|\n"
                )
            f.write(
                f"| {self.classifier_name} | {self.cv_tag}/{mode} | "
                + f"{scores[0]:.4f} | {scores[1]:.4f} | "
                + f"{scores[2]:.4f} | {scores[3]:.4f} | "
                + f"{scores[4]:.4f} | {scores[5]:.4f} |\n"
            )
