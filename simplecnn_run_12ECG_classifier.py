#!/usr/bin/env python
# Using the SimpleCNN model

import math

import numpy as np
import torch

from models.simple_cnn import SimpleCNN

# FP = "experiments/PhysioNet2020/SimpleCNN/checkpoints/model_best.pth.tar"
FP = "model_best.pth.tar"
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


def load_12ECG_model():
    """
    Load the ECG model from disk
    """
    use_cuda = torch.cuda.is_available()
    model = SimpleCNN()
    if use_cuda:
        model = model.cuda()

    try:
        checkpoint = torch.load(FP)
    except RuntimeError:
        checkpoint = torch.load(FP, map_location=torch.device("cpu"))

    sd = dict(
        (k.split("module.")[-1], v) for (k, v) in checkpoint["state_dict"].items()
    )

    model.load_state_dict(sd)
    model.eval()
    return model
