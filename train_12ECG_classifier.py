#!/usr/bin/env python

import json
import os
import time
from glob import glob

import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_extractor import hea_fp_to_np_array, structured_np_array_to_features
from util.evaluate_12ECG_score import load_table
from util.evaluation_helper import train_evaluate_score_batch_helper
from util.elapsed_timer import ElapsedTimer


def train_12ECG_classifier(
    input_directory,
    output_directory,
    data_cache_fp=".data_cache.sav",
    early_stopping_rounds=20,
    test_size=0.2,
    weights_file="weights.csv",
    experiments_to_run=100,
):
    print("Loading data...")
    header_files = tuple(
        glob(os.path.join(input_directory, "**/*.hea"), recursive=True)
    )

    print(f"Number of files: {len(header_files)}")
    data_cache = _generate_data_cache(data_cache_fp, header_files)

    # Load the SNOMED CT code mapping table
    with open("data/snomed_ct_dx_map.json", "r") as f:
        SNOMED_CODE_MAP = json.load(f)

    print("Loading weights...")

    rows, cols, all_weights = load_table(weights_file)
    assert rows == cols, "rows and cols mismatch"

    scored_codes = rows

    print("Training models...")

    for experiment_id in range(experiments_to_run):
        with ElapsedTimer() as timer:
            print(f"Experiment {experiment_id} start")

            # Split the data into train and evaluation sets
            data_train, data_eval = train_test_split(data_cache, test_size=test_size)

            # trainers throw an error when structured arrays passed, convert to unstructured
            raw_data_train = structured_np_array_to_features(data_train)
            raw_data_eval = structured_np_array_to_features(data_eval)

            # also store the split for analysis
            to_save_data = {
                "train_records": data_train["record_name"].tolist(),
                "eval_records": data_eval["record_name"].tolist(),
            }

            for idx_sc, sc in enumerate(scored_codes):
                _abbrv, dx = SNOMED_CODE_MAP[str(sc)]
                print(f"Training classifier for {dx} (code {sc})...")

                sc, model = _train_label_classifier(
                    sc,
                    idx_sc,
                    all_weights,
                    data_train,
                    raw_data_train,
                    data_eval,
                    raw_data_eval,
                    scored_codes,
                    early_stopping_rounds,
                )

                to_save_data[sc] = model

            _display_metrics(data_eval, raw_data_eval, data_cache, to_save_data)
            _save_experiment(output_directory, to_save_data)
        print(f"Experiment {experiment_id} took {timer.duration:.2f} seconds")


def _train_label_classifier(
    sc,
    idx_sc,
    all_weights,
    data_train,
    raw_data_train,
    data_eval,
    raw_data_eval,
    scored_codes,
    early_stopping_rounds,
):
    label_weights = all_weights[idx_sc]

    train_labels, train_weights = _determine_sample_weights(
        data_train, scored_codes, label_weights
    )

    eval_labels, eval_weights = _determine_sample_weights(
        data_eval, scored_codes, label_weights
    )

    # try negative over positive https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
    pos_count = len([e for e in eval_labels if e])
    scale_pos_weight = (len(eval_labels) - pos_count) / pos_count

    model = XGBClassifier(
        booster="dart",  # gbtree, dart or gblinear
        verbosity=0,
        tree_method="gpu_hist",
        sampling_method="gradient_based",
        scale_pos_weight=scale_pos_weight,
    )

    model = model.fit(
        raw_data_train,
        train_labels,
        sample_weight=train_weights,
        eval_set=[(raw_data_train, train_labels), (raw_data_eval, eval_labels)],
        sample_weight_eval_set=[train_weights, eval_weights],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return sc, model


def _save_experiment(output_directory, to_save_data):
    print("Saving model...")

    cur_sec = int(time.time())
    filename = os.path.join(output_directory, f"finalized_model_{cur_sec}.sav")
    joblib.dump(to_save_data, filename, protocol=0)

    print(f"Saved to {filename}")


def _display_metrics(data_eval, raw_data_eval, data_cache, to_save_data):
    # Calculate the challenge related metrics on the evaluation data set
    print("Calculating challenge related metrics")
    (
        auroc,
        auprc,
        accuracy,
        f_measure,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    ) = train_evaluate_score_batch_helper(
        data_eval, raw_data_eval, data_cache, to_save_data
    )

    print(
        "AUROC | AUPRC | Accuracy | F-measure | Fbeta-measure | Gbeta-measure | Challenge metric"
    )
    print(
        f"{auroc:>5.3f} | {auprc:>5.3f} | {accuracy:>8.3f} | {f_measure:>9.3f} |"
        f" {f_beta_measure:>13.3f} | {g_beta_measure:>13.3f} | {challenge_metric:>16.3f}"
    )

    to_save_data["auroc"] = auroc
    to_save_data["auprc"] = auprc
    to_save_data["accuracy"] = accuracy
    to_save_data["f_measure"] = f_measure
    to_save_data["f_beta_measure"] = f_beta_measure
    to_save_data["g_beta_measure"] = g_beta_measure
    to_save_data["challenge_metric"] = challenge_metric


def _generate_data_cache(data_cache_fp, header_files):
    if os.path.isfile(data_cache_fp):
        print(f"Loading cached dataset from '{data_cache_fp}'")
        data_cache = joblib.load(data_cache_fp, mmap_mode="r")
    else:
        # CHUNK THE HEADER FILES
        chunk_size = 100
        chunk_idx = 0
        chunk = 0
        while chunk_idx < len(header_files):
            data_cache = np.concatenate(
                joblib.Parallel(
                    verbose=1,
                    n_jobs=-1,
                    backend="multiprocessing",  # "loky", "threading"
                )(
                    joblib.delayed(hea_fp_to_np_array)(hea_fp)
                    for hea_fp in header_files[chunk_idx : chunk_idx + chunk_size]
                )
            )
            chunk_data_cache_fp = f"{data_cache_fp}.{chunk}"
            print(f"Saving cache of dataset chunk to '{chunk_data_cache_fp}'")
            joblib.dump(data_cache, chunk_data_cache_fp)

            chunk_idx += chunk_size
            chunk += 1

        # join all of the chunks into a single numpy file
        chunk_file_paths = glob(f"{data_cache_fp}.*")
        chunk_files = []
        for chunk_fp in chunk_file_paths:
            chunk_file = joblib.load(chunk_fp, mmap_mode="r")
            chunk_files.append(chunk_file)

        print(f"Saving cache of dataset to '{data_cache_fp}'")
        data_cache = np.concatenate(chunk_files)
        joblib.dump(data_cache, data_cache_fp)

    return data_cache


def _determine_sample_weights(
    data_set, scored_codes, label_weights, weight_threshold=0.5
):
    """Using the scoring labels weights to increase the dataset size of positive labels
    """
    data_labels = []
    sample_weights = []
    for dt in data_set:
        sample_weight = None
        for dx in dt["dx"]:
            if str(dx) in scored_codes:
                _sample_weight = label_weights[scored_codes.index(str(dx))]
                if _sample_weight < weight_threshold:
                    continue
                if sample_weight is None or _sample_weight > sample_weight:
                    sample_weight = _sample_weight

        if sample_weight is None:
            # not a scored label, treat as a negative example (weight of 1)
            sample_weight = 1
            data_labels.append(False)
        else:
            data_labels.append(True)
        sample_weights.append(sample_weight)
    return data_labels, sample_weights


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
