import os

import numpy as np

from datasets import PhysioNet2020Dataset

from util.config import init_class
from util.evaluation import compute_auc, compute_beta_score

from . import BaseAgent


class ScikitLearnAgent(BaseAgent):
    """Agent for PhysioNet 2020 challenge classification experiments using Scikit Learn

    Multilabel classification problem.

    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.neighbors.KNeighborsClassifier
    sklearn.neural_network.MLPClassifier
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
    sklearn.linear_model.RidgeClassifierCV
    """

    def __init__(self, config):
        super(ScikitLearnAgent, self).__init__(config)

        # Initialize cross validation datasets
        self.data_dir = config.get(
            "data_dir", "experiments/PhysioNet2020/FeatureExtraction/out"
        )
        cross_validation = config.get("cross_validation", None)

        train_records, val_records = PhysioNet2020Dataset.split_names_cv(
            self.data_dir, endswith=".npz", **cross_validation
        )
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

        self.classifier = init_class(config.get("classifier"))
        self.logger.info(self.classifier)

    def run(self):
        def load_data_cache(train_record):
            fp = os.path.join(self.data_dir, f"{train_record}.npz")
            data = np.load(fp)

            inputs = data["inputs"]
            targets = data["target"]
            return inputs, targets

        self.logger.info("Preparing training data...")
        train_data = map(load_data_cache, self.train_records)
        inputs, targets = zip(*train_data)
        inputs = np.stack(inputs)
        targets = np.stack(targets)

        self.logger.info("Fitting classifier on training data...")
        self.classifier.fit(inputs, targets)

        self.logger.info("Calculating scores on training data...")
        self.evaluate_and_log(inputs, targets, mode="Training")

        self.logger.info("Preparing validation data...")
        data = map(load_data_cache, self.val_records)
        inputs, targets = zip(*data)
        inputs = np.stack(inputs)
        targets = np.stack(targets)

        self.logger.info("Calculating scores on validation data...")
        self.evaluate_and_log(inputs, targets, mode="Validation")

    def evaluate_and_log(self, inputs, targets, mode="Training"):
        outputs = self.classifier.predict(inputs)
        probabilities = self.classifier.predict_proba(inputs)

        if type(probabilities) == list:
            for idx in range(len(probabilities)):
                probabilities[idx] = probabilities[idx][:, 1]
            probabilities = np.stack(probabilities).T

        num_inputs, _ = inputs.shape
        train_scores = np.zeros((num_inputs, 6))
        # accuracy, f_measure, f_beta, g_beta
        beta_score = compute_beta_score(targets, outputs)
        # auroc, auprc
        auc_score = compute_auc(targets, probabilities)
        train_scores = np.array(beta_score + auc_score)
        self.logger.info(f"{mode} set mean scores:")
        self.logger.info(
            f"acc: {train_scores[0]} | f_measure: {train_scores[1]} | "
            + f"f_beta: {train_scores[2]} | g_beta: {train_scores[3]} | "
            + f"auroc: {train_scores[4]} | auprc: {train_scores[5]} "
        )
