#!/usr/bin/env python3
# Using the Scikit learn agent trained model

import math
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from util.sigproc import extract_record_features
from util.sigdat import convert_to_wfdb_record, extract_features

LABELS = ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
THRESHOLD = 0.9
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


def run_12ECG_classifier(data, header_data, classes, model):
    lead_classifiers, stack_classifier = model
    num_classes = len(classes)

    r = convert_to_wfdb_record(data, header_data)
    features = extract_features(r)

    sex = features.pop("sex")
    age = features.pop("age")
    target = features.pop("target")
    sig = features.pop("sig")

    # evaluate each individual lead classifier
    lead_probabilities = OrderedDict()
    lead_outputs = OrderedDict()
    for lead, sig_features in sig.items():
        if sig_features is None:
            continue

        lead_inputs = np.concatenate([sex.flatten(), age.flatten(), *[v.flatten() for v in sig_features.values()]])
        lead_inputs = np.reshape(lead_inputs, (1, -1))

        lead_probability = lead_classifiers[lead].predict_proba(lead_inputs)
        lead_output = lead_classifiers[lead].predict(lead_inputs)
        if type(lead_probability) == list:
            for idx in range(len(lead_probability)):
                lead_probability[idx] = lead_probability[idx][:, 1]
            lead_probability = np.stack(lead_probability).T
        lead_probabilities[lead] = lead_probability
        lead_outputs[lead] = lead_output

    # construct the stack classifier inputs
    if len(lead_probabilities) == 12:
        # use the stack classifier

        stack_inputs = [lead_probabilities[lead] for lead in LEADS]
        stack_classifier_input = np.concatenate(stack_inputs, axis=1)
        probabilities = stack_classifier.predict_proba(stack_classifier_input)
        if type(probabilities) == list:
            for idx in range(len(probabilities)):
                probabilities[idx] = probabilities[idx][:, 1]
            probabilities = np.stack(probabilities).T
        outputs = stack_classifier.predict(stack_classifier_input)
    else:
        # use consensus among lead outputs
        probabilities = np.mean(np.stack(list(a.flatten() for a in lead_probabilities.values())), axis=0)
        votes = np.sum(np.stack(list(lead_outputs.values())).squeeze(), axis=0)
        if np.max(votes) > 0:
            outputs = (votes / (0.9 * np.max(votes)) >= 1).astype(int).flatten()
        else:
            # default to RBBB
            # ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
            outputs = np.zeros(9)
            outputs[6] = 1
            outputs = outputs.astype(int)

    return outputs.squeeze(), probabilities.squeeze()


def load_12ECG_model(
    fp="experiments/PhysioNet2020/Scikit_Learn/GradientBoostingClassifier/no-cv/out/",
):
    """
    Load the ECG model from disk
    """
    lc_fp = os.path.join(fp, "lead_classifiers.pkl")
    with open(lc_fp, "rb") as f:
        lead_classifiers = pickle.load(f)
    sc_fp = os.path.join(fp, "stack_classifier.pkl")
    with open(sc_fp, "rb") as f:
        stack_classifier = pickle.load(f)

    return (
        lead_classifiers,
        stack_classifier,
    )


# Classify using SimpleCNN model
def main():
    from driver import get_classes, load_challenge_data, save_challenge_predictions

    parser = ArgumentParser(
        "Run Feature Extraction + Scikit Model 12ECG classifier",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", help="Path to serialized models directory")
    parser.add_argument(
        "-i", "--input", default="Training_WFDB", help="Input directory"
    )
    parser.add_argument("-o", "--output", default="out", help="Output directory")

    args = parser.parse_args()

    input_directory = args.input
    output_directory = args.output

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if (
            os.path.isfile(os.path.join(input_directory, f))
            and not f.lower().startswith(".")
            and f.lower().endswith("mat")
        ):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    classes = get_classes(input_directory, input_files)

    # Load model.
    print("Loading 12ECG model...")
    if args.checkpoint:
        model = load_12ECG_model(fp=args.checkpoint)
    else:
        model = load_12ECG_model()

    # Iterate over files.
    print("Extracting 12ECG features...")
    num_files = len(input_files)

    with tqdm(enumerate(input_files), desc="Evaluating...", total=num_files) as t:
        for i, f in t:
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            current_label, current_score = run_12ECG_classifier(
                data, header_data, classes, model
            )
            # Save results.
            save_challenge_predictions(
                output_directory, f, current_score, current_label, classes
            )

    print(f"Done. Saved to {output_directory}")


if __name__ == "__main__":
    main()
