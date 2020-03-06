#!/usr/bin/env python
# Using the SimpleCNN model

import math
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from models.simple_cnn import SimpleCNN

from .driver import get_classes

LABELS = ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
THRESHOLD = 0.9


def run_12ECG_classifier(data, header_data, classes, model):
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # data.shape == (CHANNEL, SIGNAL)
    # SimpleCNN supports (batch, 4000, 12)
    _, seq_len = data.shape
    num_splits = math.ceil(seq_len / 4000)

    segments = np.array_split(data, num_splits, axis=1)
    for idx, segment in enumerate(segments):
        _, seq_len = segment.shape
        pad_left = 4000 - seq_len
        padded = np.pad(segment, ((0, 0), (pad_left, 0)), mode="constant")
        segments[idx] = np.transpose(padded)

    signal = np.stack(segments, axis=0)
    signal = torch.FloatTensor(signal)
    if torch.cuda.is_available():
        signal = signal.cuda()

    batch = {"signal": signal}

    out = model(batch)
    out = out.detach().cpu().numpy()

    # average out all of the sequences within the batch
    out = np.average(out, axis=0)

    # rescale outputs to min 0, max 1
    current_score = np.interp(out, (out.min(), out.max()), (0, 1))
    current_label = (current_score >= THRESHOLD) * 1

    return current_label, current_score


def load_12ECG_model(
    fp="experiments/PhysioNet2020/SimpleCNN/checkpoints/model_best.pth.tar",
):
    """
    Load the ECG model from disk
    """
    use_cuda = torch.cuda.is_available()
    model = SimpleCNN()
    if use_cuda:
        model = model.cuda()

    try:
        checkpoint = torch.load(fp)
    except RuntimeError:
        checkpoint = torch.load(fp, map_location=torch.device("cpu"))

    sd = dict(
        (k.split("module.")[-1], v) for (k, v) in checkpoint["state_dict"].items()
    )

    model.load_state_dict(sd)
    model.eval()
    return model


# Classify using SimpleCNN model
def main():
    parser = ArgumentParser(
        "Run SimpleCNN 12ECG classifier", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint")
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
    model = load_12ECG_model()

    # Iterate over files.
    print("Extracting 12ECG features...")
    num_files = len(input_files)

    with tqdm(enumerate(input_files), desc="Evaluating...") as t:
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
