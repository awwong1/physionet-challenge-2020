import json
import os
from datetime import datetime

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

from datasets import PhysioNet2020Dataset
from util.config import init_class
from util.evaluation import compute_auc, compute_beta_score

from . import BaseAgent


class ScikitLearnAgent(BaseAgent):
    """Agent for PhysioNet 2020 challenge classification experiments using Scikit Learn classifiers

    Each lead gets its own classifier, then aggregate all results into another classifier
    All classifiers must support multilabel
    """

    def __init__(self, config):
        super(ScikitLearnAgent, self).__init__(config)

        self.leads = (
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

        # Initialize cross validation datasets
        self.data_dir = config.get(
            "data_dir", "experiments/PhysioNet2020/FeatureExtraction/out"
        )
        self.output_dir = config["out_dir"]
        cross_validation = config.get("cross_validation", None)

        train_records, val_records = PhysioNet2020Dataset.split_names_cv(
            self.data_dir, endswith=".npz", **cross_validation
        )

        self.cv_tag = "cv{fold}-{val_offset}".format(**cross_validation)

        self.logger.info(
            "Training on %d records (%s, ..., %s)",
            len(train_records),
            train_records[0],
            train_records[-1],
        )
        self.logger.info(
            "Validating on %d records (%s, ..., %s)",
            len(val_records),
            val_records[0],
            val_records[-1],
        )
        self.train_records = train_records
        self.val_records = val_records

        # initialize classifiers for all 12 leads
        self.lead_classifiers = dict(
            (k, init_class(config.get("lead_classifier"))) for k in self.leads
        )
        for k, v in self.lead_classifiers.items():
            self.logger.info(f"{k}: {v}")
        # initialize the meta classifiers (uses lead classifiers output as input)
        self.stack_classifier = init_class(config.get("stack_classifier"))
        self.logger.info(f"stack: {self.stack_classifier}")

    def run(self):
        def load_data_cache(train_record):
            fp = os.path.join(self.data_dir, f"{train_record}.npz")
            data = np.load(fp)
            return (
                {
                    "I": data["I"],
                    "II": data["II"],
                    "III": data["III"],
                    "aVR": data["aVR"],
                    "aVL": data["aVL"],
                    "aVF": data["aVF"],
                    "V1": data["V1"],
                    "V2": data["V2"],
                    "V3": data["V3"],
                    "V4": data["V4"],
                    "V5": data["V5"],
                    "V6": data["V6"],
                },
                data["target"],
            )

        self.logger.info("Preparing training data...")
        start = datetime.now()
        train_data = map(load_data_cache, self.train_records)
        inputs, targets = zip(*train_data)
        lead_inputs = {}
        for lead in self.leads:
            lead_inputs[lead] = np.stack(tuple(inp[lead] for inp in inputs))
        targets = np.stack(targets)
        self.logger.info(f"Took {datetime.now() - start}\n")

        # train each individual lead classifier
        stack_inputs = []
        for lead in self.leads:
            shape = lead_inputs[lead].shape
            self.logger.info(
                f"Fitting lead {lead} classifier on training data {shape}..."
            )
            start = datetime.now()
            self.lead_classifiers[lead].fit(lead_inputs[lead], targets)
            self.logger.info(f"Took {datetime.now() - start}\n")

            try:
                stack_input = self.lead_classifiers[lead].predict_proba(
                    lead_inputs[lead]
                )
            except AttributeError as e:
                stack_input = self.lead_classifiers[lead].predict(lead_inputs[lead])

            if type(stack_input) == list:
                for idx in range(len(stack_input)):
                    stack_input[idx] = stack_input[idx][:, 1]
                stack_input = np.stack(stack_input).T
            stack_inputs.append(stack_input)

        # construct the data for the meta/stack classifier
        stack_inputs = np.concatenate(stack_inputs, axis=1)
        shape = stack_inputs.shape
        self.logger.info(f"Fitting stack classifier on training data {shape}...")
        start = datetime.now()
        self.stack_classifier.fit(stack_inputs, targets)
        self.logger.info(f"Took {datetime.now() - start}\n")

        self.logger.info("Calculating scores on training data...")
        start = datetime.now()
        self.evaluate_and_log(lead_inputs, targets, mode="Training")
        self.logger.info(f"Took {datetime.now() - start}\n")

        self.logger.info("Preparing validation data...")
        start = datetime.now()
        data = map(load_data_cache, self.val_records)
        inputs, targets = zip(*data)
        lead_inputs = {}
        for lead in self.leads:
            lead_inputs[lead] = np.stack(tuple(inp[lead] for inp in inputs))
        targets = np.stack(targets)
        self.logger.info(f"Took {datetime.now() - start}\n")

        self.logger.info("Calculating scores on validation data...")
        start = datetime.now()
        self.evaluate_and_log(lead_inputs, targets, mode="Validation")
        self.logger.info(f"Took {datetime.now() - start}\n")

    def finalize(self):
        return 
        # TODO: parameters do not actually correspond to classifier state, this needs to become pickle'd

        # classifiers = [("stack", self.stack_classifier), *self.lead_classifiers.items()]
        # for name, classifier in classifiers:
        #     params = classifier.get_params()
        #     try:
        #         wp = os.path.join(self.output_dir, f"{name}_params.json")
        #         with open(wp, "w") as f:
        #             json.dump(params, f)
        #             self.logger.info(f"Saved classifier parameters to {wp}")
        #     except TypeError:
        #         # TypeError: Object of type 'ndarray' is not JSON serializable
        #         wp = os.path.join(self.output_dir, f"{name}_params.npy")
        #         np.save(wp, params)
        #         self.logger.info(f"Saved classifier parameters to {wp}")

    def evaluate_and_log(self, lead_inputs, targets, mode="Training"):
        # evaluate each individual lead classifier
        stack_inputs = []
        for lead in self.leads:
            try:
                stack_input = self.lead_classifiers[lead].predict_proba(
                    lead_inputs[lead]
                )
            except AttributeError as e:
                stack_input = self.lead_classifiers[lead].predict(lead_inputs[lead])

            if type(stack_input) == list:
                for idx in range(len(stack_input)):
                    stack_input[idx] = stack_input[idx][:, 1]
                stack_input = np.stack(stack_input).T
            stack_inputs.append(stack_input)

        stack_inputs = np.concatenate(stack_inputs, axis=1)
        outputs = self.stack_classifier.predict(stack_inputs)

        # if output shape is not two dimensional, expand
        if len(outputs.shape) == 1:
            lb = LabelBinarizer()
            lb.fit(outputs)
            outputs = lb.transform(outputs)

        try:
            probabilities = self.stack_classifier.predict_proba(stack_inputs)
        except AttributeError as e:
            probabilities = outputs.astype(np.float)

        if type(probabilities) == list:
            for idx in range(len(probabilities)):
                probabilities[idx] = probabilities[idx][:, 1]
            probabilities = np.stack(probabilities).T

        num_inputs, _ = stack_inputs.shape
        train_scores = np.zeros((num_inputs, 6))
        # accuracy, f_measure, f_beta, g_beta
        beta_score = compute_beta_score(targets, outputs)
        # auroc, auprc
        auc_score = compute_auc(targets, probabilities)
        train_scores = np.array(beta_score + auc_score)
        self.logger.info(f"{self.cv_tag}/{mode} mean scores:")
        self.logger.info(
            f"acc: {train_scores[0]} | f_measure: {train_scores[1]} | "
            + f"f_beta: {train_scores[2]} | g_beta: {train_scores[3]} | "
            + f"auroc: {train_scores[4]} | auprc: {train_scores[5]} "
        )

        wp = os.path.join(self.output_dir, "scores.txt")
        file_exists = os.path.isfile(wp)
        with open(wp, "a") as f:
            if not file_exists:
                f.write(
                    "| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |\n"
                    + "|---------|----------|-----------|--------|--------|-------|-------|\n"
                )
            f.write(
                f"| {self.cv_tag}/{mode} | "
                + f"{train_scores[0]:.3f} | {train_scores[1]:.3f} | "
                + f"{train_scores[2]:.3f} | {train_scores[3]:.3f} | "
                + f"{train_scores[4]:.3f} | {train_scores[5]:.3f} |\n"
            )
