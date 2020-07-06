#!/usr/bin/env python

import os
import re
from glob import glob

import joblib
import numpy as np
import wfdb
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from feature_extractor import get_structured_lead_features
from get_12ECG_features import get_12ECG_features


def hea_fp_to_np_array(hea_fp):
    """Read a .hea file, convert it into a structured numpy array containing all ECG comment metadata and signal features
    """
    record_name = hea_fp.split(".hea")[0]
    r = wfdb.rdrecord(record_name)
    signal = r.p_signal
    seq_len, num_leads = signal.shape

    # Comment derived features
    dx = []  # target label
    age = float("nan")
    sex = float("nan")

    for comment in r.comments:
        dx_grp = re.search(r"Dx: (?P<dx>.*)$", comment)
        if dx_grp:
            raw_dx = dx_grp.group("dx").split(",")
            for dxi in raw_dx:
                snomed_code = int(dxi)
                dx.append(snomed_code)
            continue

        age_grp = re.search(r"Age: (?P<age>.*)$", comment)
        if age_grp:
            age = float(age_grp.group("age"))
            if not np.isfinite(age):
                age = float("nan")
            continue

        sx_grp = re.search(r"Sex: (?P<sx>.*)$", comment)
        if sx_grp:
            if sx_grp.group("sx").upper().startswith("F"):
                sex = 1.0
            elif sx_grp.group("sx").upper().startswith("M"):
                sex = 0.0
            continue

    # Base structure of numpy array
    data = [record_name, seq_len, r.fs, age, sex, tuple(dx)]
    dtype = [
        ("record_name", np.unicode_, 50),
        ("seq_len", "f4"),
        ("sampling_rate", "f4"),
        ("age", "f4"),
        ("sex", "f4"),
        ("dx", np.object),
    ]

    # Signal derived features
    for lead_idx in range(num_leads):
        lead_sig = r.p_signal[:, lead_idx]
        lead_data, lead_dtype = get_structured_lead_features(
            lead_sig,
            sampling_rate=r.fs,
            lead_name=r.sig_name[lead_idx]
        )
        data += lead_data
        dtype += lead_dtype

    return np.array([tuple(data), ], dtype=np.dtype(dtype))


def train_12ECG_classifier(input_directory, output_directory, data_cache_fp=".data_cache.sav"):
    print("Loading data...")
    header_files = tuple(
        glob(os.path.join(input_directory, "**/*.hea"), recursive=True)
    )

    print(f"Number of files: {len(header_files)}")

    if os.path.isfile(data_cache_fp):
        print(f"Loading cached dataset from '{data_cache_fp}'")
        data_cache = joblib.load(data_cache_fp)
    else:
        data_cache = np.concatenate(
            joblib.Parallel(verbose=1, n_jobs=-1)(
                joblib.delayed(hea_fp_to_np_array)(hea_fp) for hea_fp in header_files
            )
        )
        print(f"Saving cache of dataset to '{data_cache_fp}'")
        joblib.dump(data_cache, data_cache_fp)

    classes = sorted(set([dxi for dxs in data_cache["dx"] for dxi in dxs]))
    num_classes = len(classes)
    num_files = len(data_cache)

    print("Skipping training model.")
    return

    # Train model.
    print("Training model...")

    features = list()
    labels = list()

    for i in range(num_files):
        recording = recordings[i]
        header = headers[i]

        tmp = get_12ECG_features(recording, header)
        features.append(tmp)

        for l in header:
            if l.startswith("#Dx:"):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(" ")
                for arr in arrs[1].split(","):
                    class_index = classes.index(
                        arr.rstrip()
                    )  # Only use first positive index
                    labels_act[class_index] = 1
        labels.append(labels_act)

    features = np.array(features)
    labels = np.array(labels)

    # Replace NaN values with mean values
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    # Train the classifier
    model = RandomForestClassifier().fit(features, labels)

    # Save model.
    print("Saving model...")

    final_model = {"model": model, "imputer": imputer}

    filename = os.path.join(output_directory, "finalized_model.sav")
    joblib.dump(final_model, filename, protocol=0)


# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, "r") as f:
        header = f.readlines()
    mat_file = header_file.replace(".hea", ".mat")
    x = loadmat(mat_file)
    recording = np.asarray(x["val"], dtype=np.float64)
    return recording, header


# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, "r") as f:
            for l in f:
                if l.startswith("#Dx"):
                    tmp = l.split(": ")[1].split(",")
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)
