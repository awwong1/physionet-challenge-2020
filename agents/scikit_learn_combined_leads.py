import json
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

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

        self.classifier = init_class(config["classifier"])
        self.classifier_name = config["classifier"]["name"].split(".")[-1]

        self.variance_threshold = config.get("variance_threshold", None)
        self.use_multioutput = config.get("use_multioutput", False)
        self.classifier_chain_order = config.get("classifier_chain_order", None)

        if self.variance_threshold:
            self.classifier = Pipeline(
                [
                    VarianceThreshold(
                        threshold=(
                            self.variance_threshold * (1 - self.variance_threshold)
                        )
                    ),
                    self.classifier,
                ]
            )
        if self.use_multioutput:
            self.classifier = MultiOutputClassifier(self.classifier, n_jobs=-1)
        if self.classifier_chain_order:
            self.classifier = ClassifierChain(
                self.classifier, order=self.classifier_chain_order
            )
        self.logger.info(self.classifier)

    def run(self):
        inputs, targets = self.prepare_data(self.train_records, mode="Training")
        self.logger.info(f"Fit training data to classifier {inputs.shape}...")
        start = datetime.now()
        self.classifier.fit(inputs, targets)
        self.logger.info(f"Took {datetime.now() - start}")

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

        # raw_data = np.concatenate([data[k] for k in ScikitLearnAgent.LEADS])
        # only record sex (2) and age (1) once
        raw_data = np.concatenate(
            [data["I"][:3]] + [data[k][3:] for k in ScikitLearnAgent.LEADS]
        )

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
        try:
            probabilities = np.stack([prob[:, 1] for prob in probabilities]).T
        except IndexError:
            probabilities = np.stack(probabilities)

        # accuracy, f_measure, f_beta, g_beta
        beta_score = compute_beta_score(targets, outputs)
        # auroc, auprc
        auc_score = compute_auc(targets, probabilities)
        scores = np.array(beta_score + auc_score)
        self.logger.info(f"{self.cv_tag}/{mode} mean scores:")

        pipeline = ""
        if self.variance_threshold:
            pipeline += f"VarianceThreshold({self.variance_threshold})"
        if self.use_multioutput:
            pipeline += "MultiOutputClassifier"
        if self.classifier_chain_order:
            pipeline += f"ClassifierChain(order={self.classifier_chain_order})"
        if not pipeline:
            pipeline = "None"

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
