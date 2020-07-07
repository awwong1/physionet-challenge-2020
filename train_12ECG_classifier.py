#!/usr/bin/env python

import os
import json
import re
import time
from glob import glob

import joblib
import numpy as np
import wfdb
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_extractor import get_structured_lead_features


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
            lead_sig, sampling_rate=r.fs, lead_name=r.sig_name[lead_idx]
        )
        data += lead_data
        dtype += lead_dtype

    return np.array([tuple(data),], dtype=np.dtype(dtype))


def train_12ECG_classifier(
    input_directory,
    output_directory,
    data_cache_fp=".data_cache.sav",
    early_stopping_rounds=20,
    test_size=0.2,
    scored_codes=[
        270492004,
        164889003,
        164890007,
        426627000,
        713427006,  # A: 713427006 and 59118001
        713426002,
        445118002,
        39732003,
        164909002,
        251146004,
        698252002,
        10370003,
        284470004,  # B: 284470004 and 63593006
        427172004,  # C: 427172004 and 17338001
        164947007,
        111975006,
        164917005,
        47665007,
        59118001,  # A: 713427006 and 59118001
        427393009,
        426177001,
        426783006,
        427084000,
        63593006,  # B: 284470004 and 63593006
        164934002,
        59931005,
        17338001,  # C: 427172004 and 17338001
    ],
):
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

    # classes = sorted(set([dxi for dxs in data_cache["dx"] for dxi in dxs]))
    # num_classes = len(classes)

    to_features = [
        n for n in data_cache.dtype.names if n not in ("dx", "record_name", "seq_len")
    ]

    # Split the data into train and evaluation sets
    data_train, data_eval = train_test_split(data_cache, test_size=test_size)

    # trainers throw an error when structured arrays passed, convert to unstructured
    raw_data_train = data_train[to_features].copy()
    raw_data_train = raw_data_train.tolist()
    raw_data_train = np.array(raw_data_train)

    raw_data_eval = data_eval[to_features].copy()
    raw_data_eval = raw_data_eval.tolist()
    raw_data_eval = np.array(raw_data_eval)

    # also store the split for analysis
    to_save_data = {
        "train_records": data_train["record_name"].tolist(),
        "eval_records": data_eval["record_name"].tolist(),
    }

    # Load the SNOMED CT code mapping table
    with open("data/snomed_ct_dx_map.json", "r") as f:
        SNOMED_CODE_MAP = json.load(f)

    # Train model.
    print("Training models...")
    for sc in scored_codes:

        _abbrv, dx = SNOMED_CODE_MAP[str(sc)]

        # hardcoded duplicate classifiers based on label scoring weights
        dsc = None
        if sc == 713427006:
            # A: 713427006 and 59118001
            dsc = 59118001
        elif sc == 284470004:
            # B: 284470004 and 63593006
            dsc = 63593006
        elif sc == 427172004:
            # C: 427172004 and 17338001
            dsc = 17338001

        if dsc is not None:
            _, ddx = SNOMED_CODE_MAP[str(dsc)]
            print(f"Skipping {dx} (code {sc}), covered by {ddx} (code {dsc})")
            continue

        print(f"Training classifier for {dx} (code {sc})...")

        isc = None
        if sc == 59118001:
            isc = 713427006
        elif sc == 63593006:
            isc = 284470004
        elif sc == 17338001:
            isc = 427172004

        if isc is not None:
            _, idx = SNOMED_CODE_MAP[str(isc)]
            print(f"Including {idx} (code {isc})")

        train_labels = []
        for dt in data_train:
            pos = (sc in dt["dx"]) or (isc is not None and isc in dt["dx"])
            train_labels.append(pos)
        eval_labels = []
        for dt in data_eval:
            pos = (sc in dt["dx"]) or (isc is not None and isc in dt["dx"])
            eval_labels.append(pos)

        # default
        # scale_pos_weight = 1

        # try negative over positive https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
        pos_count = len([e for e in eval_labels if e])
        scale_pos_weight = (len(eval_labels) - pos_count) / pos_count

        model = XGBClassifier(
            booster="gbtree",  # gbtree, dart or gblinear
            verbosity=2,
            tree_method="gpu_hist",
            sampling_method="gradient_based",
            scale_pos_weight=scale_pos_weight,
        )
        model = model.fit(
            raw_data_train,
            train_labels,
            eval_set=[(raw_data_train, train_labels), (raw_data_eval, eval_labels)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=True,
        )

        to_save_data[sc] = model

    # Save model.
    print("Saving model...")

    cur_sec = int(time.time())

    filename = os.path.join(output_directory, f"finalized_model_{cur_sec}.sav")
    joblib.dump(to_save_data, filename, protocol=0)

    print(f"Saved to {filename}")


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
