import json
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.multioutput import MultiOutputClassifier
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

        setup_lead_pipeline_feature_selection = config.get(
            "lead_pipeline_feature_selection", None
        )

        # determine if multioutput classification plug is required
        is_multioutput = self.check_multioutput(init_class(config["lead_classifier"]))

        self.lead_classifiers = {}
        for k in ScikitLearnAgent.LEADS:
            l_classifier = init_class(config["lead_classifier"])
            if not is_multioutput:
                l_classifier = MultiOutputClassifier(l_classifier)
            if setup_lead_pipeline_feature_selection:
                l_classifier = Pipeline(
                    [
                        (
                            "feature_selection",
                            SelectFromModel(
                                init_class(setup_lead_pipeline_feature_selection)
                            ),
                        ),
                        ("lead_classification", l_classifier,),
                    ]
                )
            self.lead_classifiers[k] = l_classifier

        self.lead_classifier_name = config["lead_classifier"]["name"].split(".")[-1]

        for k, v in self.lead_classifiers.items():
            self.logger.info(f"{k}: {v}")
            self.logger.info("same for other 11 leads...")
            break
        # initialize the meta classifiers (uses lead classifiers output as input)
        self.stack_classifier = init_class(config.get("stack_classifier"))
        self.logger.info(f"stack: {self.stack_classifier}")

    def run(self):
        lead_inputs, lead_targets, lead_keys = self.prepare_data(
            self.train_records, mode="Training"
        )

        stack_inputs = {}
        for lead in ScikitLearnAgent.LEADS:
            shape = lead_inputs[lead].shape
            self.logger.info(
                f"Fitting lead {lead} classifier on training data {shape}..."
            )
            start = datetime.now()
            self.lead_classifiers[lead].fit(lead_inputs[lead], lead_targets[lead])
            self.logger.info(f"Took {datetime.now() - start}")

            self.logger.info(
                f"Evaluating lead {lead} classifier on training data {shape}..."
            )
            start = datetime.now()
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
            stack_inputs[lead] = stack_input
            self.logger.info(f"Took {datetime.now() - start}")

        # construct the stack classifier inputs
        stack_classifier_inputs = []
        stack_classifier_targets = []
        for idx in range(len(self.train_records)):
            # if the index is in all lead keys, it can be used as input
            missing_lead = False
            lead_idxs = {}
            for lead in self.LEADS:
                try:
                    lead_idxs[lead] = lead_keys[lead].index(idx)
                except ValueError:
                    missing_lead = True
                    break

            if missing_lead:
                continue

            stack_classifier_input = []
            for lead, lead_idx in lead_idxs.items():
                stack_classifier_input.append(stack_inputs[lead][lead_idx])
                if lead == self.LEADS[0]:
                    stack_classifier_targets.append(lead_targets[lead][lead_idx])
            stack_classifier_input = np.concatenate(stack_classifier_input)
            stack_classifier_inputs.append(stack_classifier_input)
        stack_classifier_inputs = np.stack(stack_classifier_inputs, axis=0)
        stack_classifier_targets = np.stack(stack_classifier_targets, axis=0)

        self.logger.info(
            f"Fitting stack classifier on training data {stack_classifier_inputs.shape}..."
        )
        start = datetime.now()
        self.stack_classifier.fit(stack_classifier_inputs, stack_classifier_targets)
        self.logger.info(f"Took {datetime.now() - start}")

        self.logger.info("Calculating scores on training data...")
        start = datetime.now()
        self.evaluate_and_log(lead_inputs, lead_targets, lead_keys, mode="Training")
        self.logger.info(f"Took {datetime.now() - start}")

        if self.val_records:
            lead_inputs, lead_targets, lead_keys = self.prepare_data(
                self.val_records, mode="Validation"
            )

            self.logger.info("Calculating scores on validation data...")
            start = datetime.now()
            self.evaluate_and_log(
                lead_inputs, lead_targets, lead_keys, mode="Validation"
            )
            self.logger.info(f"Took {datetime.now() - start}")

    def finalize(self):
        if self.serialize_model:
            lc_fp = os.path.join(self.output_dir, "lead_classifiers.pkl")
            with open(lc_fp, "wb") as f:
                pickle.dump(self.lead_classifiers, f)
            self.logger.info(f"Saved lead classifiers to: {lc_fp}")
            sc_fp = os.path.join(self.output_dir, "stack_classifier.pkl")
            with open(sc_fp, "wb") as f:
                pickle.dump(self.stack_classifier, f)
            self.logger.info(f"saved stack classifier to: {sc_fp}")

    def prepare_data(self, records, mode="Training"):
        self.logger.info(f"Preparing {mode} data...")
        start = datetime.now()
        train_data = map(self.load_data_cache, records)
        inputs, targets = zip(*train_data)

        lead_keys = {}
        lead_inputs = {}
        lead_targets = {}
        for lead in ScikitLearnAgent.LEADS:
            lead_key = []
            lead_input = []
            lead_target = []
            for idx, inp in enumerate(inputs):
                if inp[lead] is not None:
                    lead_input.append(inp[lead])
                    lead_target.append(targets[idx])
                    lead_key.append(idx)
            lead_inputs[lead] = np.stack(lead_input)
            lead_targets[lead] = np.stack(lead_target)
            lead_keys[lead] = tuple(lead_key)
        self.logger.info(f"Took {datetime.now() - start}")
        return lead_inputs, lead_targets, lead_keys

    def load_data_cache(self, train_record):
        fp = os.path.join(self.data_dir, f"{train_record}.npz")
        data = np.load(fp)
        return (
            {
                "I": data.get("I", None),
                "II": data.get("II", None),
                "III": data.get("III", None),
                "aVR": data.get("aVR", None),
                "aVL": data.get("aVL", None),
                "aVF": data.get("aVF", None),
                "V1": data.get("V1", None),
                "V2": data.get("V2", None),
                "V3": data.get("V3", None),
                "V4": data.get("V4", None),
                "V5": data.get("V5", None),
                "V6": data.get("V6", None),
            },
            data["target"],
        )

    def evaluate_and_log(self, lead_inputs, lead_targets, lead_keys, mode="Training"):
        # evaluate each individual lead classifier
        stack_inputs = {}
        for lead in ScikitLearnAgent.LEADS:
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
            stack_inputs[lead] = stack_input

        # construct the stack classifier inputs
        stack_classifier_inputs = []
        consensus_indicies = []
        stack_indicies = []

        records = self.train_records
        if mode != "Training":
            records = self.val_records
        for idx in range(len(records)):
            # if the index is in all lead keys, it can be used as input
            missing_lead = False
            lead_idxs = {}
            for lead in self.LEADS:
                try:
                    lead_idxs[lead] = lead_keys[lead].index(idx)
                except ValueError:
                    missing_lead = True
                    break

            if missing_lead:
                consensus_indicies.append(idx)
                continue
            else:
                stack_indicies.append(idx)

            stack_classifier_input = []
            for lead, lead_idx in lead_idxs.items():
                stack_classifier_input.append(stack_inputs[lead][lead_idx])
            stack_classifier_input = np.concatenate(stack_classifier_input)
            stack_classifier_inputs.append(stack_classifier_input)
        stack_classifier_inputs = np.stack(stack_classifier_inputs, axis=0)

        # Get the stack outputs and stack probabilities
        stack_outputs = self.stack_classifier.predict(stack_classifier_inputs)
        try:
            stack_probabilities = np.stack(
                [
                    tup[:, 1]
                    for tup in self.stack_classifier.predict_proba(
                        stack_classifier_inputs
                    )
                ],
                axis=1,
            )
        except AttributeError as e:
            stack_probabilities = stack_outputs.astype(np.float)

        # handle all missing lead indicies as average/consensus of the leads
        lead_outputs = []
        lead_probabilities = []
        for consensus_idx in consensus_indicies:
            lead_output = []
            lead_probability = []
            for lead in self.LEADS:
                if consensus_idx in lead_keys[lead]:
                    lead_idx = lead_keys[lead].index(consensus_idx)
                    lead_output.append(
                        self.lead_classifiers[lead].predict(
                            np.reshape(lead_inputs[lead][lead_idx], (1, -1))
                        )
                    )
                    lead_probability.append(
                        np.concatenate(
                            [
                                tup[:, 1]
                                for tup in self.lead_classifiers[lead].predict_proba(
                                    np.reshape(lead_inputs[lead][lead_idx], (1, -1))
                                )
                            ]
                        )
                    )

            # average the probabilities, consensus the lead_outputs (90% choose, mark as 1)
            lead_probabilities.append(np.mean(lead_probability, axis=0).flatten())
            # if all zeros, default to RBBB
            votes = np.sum(lead_output, axis=0)
            if np.max(votes) > 0:
                lead_outputs.append(
                    (votes / (0.9 * np.max(votes)) >= 1).astype(int).flatten()
                )
            else:
                # ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
                stub = np.zeros(9)
                stub[6] = 1
                stub = stub.astype(int)
                lead_outputs.append(stub)

        # merge the stack and lead values together
        outputs = []
        probabilities = []
        targets = []
        for idx in range(len(records)):
            if idx in stack_indicies:
                s_idx = stack_indicies.index(idx)
                outputs.append(stack_outputs[s_idx].flatten())
                probabilities.append(stack_probabilities[s_idx].flatten())
            else:
                # kludge in the consensus values
                c_idx = consensus_indicies.index(idx)
                outputs.append(lead_outputs[c_idx].flatten())
                probabilities.append(lead_probabilities[c_idx].flatten())

            # append in the target values
            for lead in self.LEADS:
                if idx in lead_keys[lead]:
                    l_idx = lead_keys[lead].index(idx)
                    targets.append(lead_targets[lead][l_idx].flatten())
                    break

        outputs = np.stack(outputs)
        probabilities = np.stack(probabilities)
        targets = np.stack(targets)

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
                    "| Lead Classifier | Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |\n"
                    + "|-----------------|---------|----------|-----------|--------|--------|-------|-------|\n"
                )
            f.write(
                f"| {self.lead_classifier_name} | {self.cv_tag}/{mode} | "
                + f"{scores[0]:.4f} | {scores[1]:.4f} | "
                + f"{scores[2]:.4f} | {scores[3]:.4f} | "
                + f"{scores[4]:.4f} | {scores[5]:.4f} |\n"
            )

    def check_multioutput(self, classifier):
        X = np.random.random((10, 50))
        y = np.random.randint(2, size=(10, 9))

        is_multioutput = True

        try:
            classifier.fit(X, y)
            out = classifier.predict(X)
            assert out.shape == (10, 9)
            pass
        except Exception as e:
            self.logger.info("Wrapping in MultiOutputClassifier...")
            is_multioutput = False
        return is_multioutput
