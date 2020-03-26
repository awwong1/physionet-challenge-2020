#!/usr/bin/env python
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
    features = extract_record_features(
        data, header_data, template_resample=60, include_fourier=True
    )

    sex = features.pop("sex")
    age = features.pop("age")
    target = features.pop("target")

    sig = features.pop("sig")
    # split out each lead as as its own feature array
    sig_concat = OrderedDict()
    for sig_name, sig_features in sig.items():
        sig_concat[sig_name] = np.concatenate(
            [
                sex.flatten(),
                age.flatten(),
                *[v.flatten() for v in sig_features.values()],
            ]
        )

    # run each separate lead in their respective lead classifier
    stack_inputs = []
    for sig_name in LEADS:
        sig_concat_features = sig_concat[sig_name]
        try:
            stack_input = lead_classifiers[sig_name].predict_proba(sig_concat_features.reshape(1, -1))
        except AttributeError as e:
            stack_input = self.lead_classifiers[sig_name].predict(sig_concat_features.reshape(1, -1))

        if type(stack_input) == list:
            for idx in range(len(stack_input)):
                stack_input[idx] = stack_input[idx][:, 1]
            stack_input = np.stack(stack_input).T
        stack_inputs.append(stack_input)


    dims = set([len(si.shape) for si in stack_inputs])
    assert len(dims) == 1, "stack dimensions must be equal"
    dim = dims.pop()
    if dim == 2:
        stack_inputs = np.concatenate(stack_inputs, axis=1)
    elif dim == 1:
        stack_inputs = np.stack(stack_inputs).T

    outputs = stack_classifier.predict(stack_inputs)

    # if output shape is not two dimensional, expand
    if len(outputs.shape) == 1:
        lb = LabelBinarizer()
        lb.fit(outputs)
        outputs = lb.transform(outputs)

    outputs = outputs.astype(np.int)

    try:
        probabilities = stack_classifier.predict_proba(stack_inputs)
    except AttributeError as e:
        probabilities = outputs.astype(np.float)

    if type(probabilities) == list:
        for idx in range(len(probabilities)):
            probabilities[idx] = probabilities[idx][:, 1]
        probabilities = np.stack(probabilities).T

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
