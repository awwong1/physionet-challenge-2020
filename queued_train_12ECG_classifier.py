import csv
import json
import multiprocessing
import os
import queue
from datetime import datetime, timedelta
from glob import glob
from time import time

import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from neurokit2_parallel import (
    ECG_LEAD_NAMES,
    KEYS_INTERVALRELATED,
    KEYS_TSFRESH,
    wfdb_record_to_feature_dataframe,
)
from util.log import configure_logging
from util.evaluate_12ECG_score import load_table, is_number
from util.evaluation_helper import evaluate_score_batch
from util.elapsed_timer import ElapsedTimer


def _get_fieldnames():
    field_names = ["header_file", "age", "sex"]
    for lead_name in ECG_LEAD_NAMES:
        for key in KEYS_INTERVALRELATED:
            field_names.append(f"{lead_name}_{key}")
        for key in KEYS_TSFRESH:
            hb_key = f"hb__{key}"
            field_names.append(f"{lead_name}_{hb_key}")
        for key in KEYS_TSFRESH:
            sig_key = f"sig__{key}"
            field_names.append(f"{lead_name}_{sig_key}")
    return field_names


def feat_extract_process(
    input_queue: multiprocessing.JoinableQueue,
    output_queue: multiprocessing.JoinableQueue,
):
    while True:
        try:
            header_file_path = input_queue.get_nowait()
            input_queue.task_done()
        except queue.Empty:
            # When the input queue is empty, worker process terminates
            break
        r = wfdb.rdrecord(header_file_path.rsplit(".hea")[0])

        record_features, dx = wfdb_record_to_feature_dataframe(r)

        # turn dataframe record_features into dict flatten out the values (one key to one row)
        ecg_features = dict((k, v[0]) for (k, v) in record_features.to_dict().items())
        output_queue.put((header_file_path, ecg_features, dx))


def train_12ECG_classifier(
    input_directory,
    output_directory,
    labels_fp="dxs.txt",
    features_fp="features.csv",
    weights_file="evaluation-2020/weights.csv",
    early_stopping_rounds=20,
    experiments_to_run=1,
):
    logger = configure_logging()

    logger.info("Loading feature extraction result...")
    # check how many files have been processed already, allows feature extraction to be resumable
    mapped_records = {}
    if os.path.isfile(labels_fp):
        with open(labels_fp, mode="r", newline="\n") as labelfile:
            for line in labelfile.readlines():
                header_file_path, dxs = json.loads(line)
                mapped_records[header_file_path] = dxs
        logger.info(f"Loaded {len(mapped_records)} from prior run.")
    else:
        logger.info("No prior feature extraction step performed.")
        with open(labels_fp, mode="w"):
            # initialize the file
            pass

    logger.info("Finding input files...")
    process_header_files = tuple(
        hfp
        for hfp in sorted(
            glob(os.path.join(input_directory, "**/*.hea"), recursive=True)
        )
        if hfp not in mapped_records
    )

    logger.info("Number of ECG records to process: %d", len(process_header_files))

    num_cpus = len(os.sched_getaffinity(0))
    logger.info("Number of available CPUs: %d", num_cpus)

    # Setup & populate input queue, then initialize output queue
    input_queue = multiprocessing.JoinableQueue()
    for header_file in process_header_files:
        input_queue.put_nowait(header_file)
    output_queue = multiprocessing.JoinableQueue()

    # all CPUs used for feature extraction
    num_feature_extractor_procs = max(num_cpus, 1)
    feature_extractor_procs = []
    killed_extractor_procs = []
    for _ in range(num_feature_extractor_procs):
        p = multiprocessing.Process(
            target=feat_extract_process, args=(input_queue, output_queue)
        )
        p.start()
        feature_extractor_procs.append(p)

    # main process used for concatenating features
    processed_files_counter = 0
    out_start = datetime.now()
    out_log = None
    fieldnames = _get_fieldnames()

    # initialize the header if the file does not exist
    if not os.path.isfile(features_fp):
        with open(features_fp, "w", newline="\n") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(features_fp, "a", newline="\n") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        with open(labels_fp, "a") as labelfile:
            while True:
                try:
                    header_file_path, f_dict, dxs = output_queue.get(True, 0.1)
                    labelfile.write(json.dumps((header_file_path, dxs)) + "\n")
                    labelfile.flush()
                    f_dict["header_file"] = header_file_path
                    writer.writerow(f_dict)
                    output_queue.task_done()
                    processed_files_counter += 1
                except queue.Empty:
                    # When the output queue is empty and all workers are terminated
                    # all files have been processed
                    if all(not p.is_alive() for p in feature_extractor_procs):
                        break

                    num_feature_extractor_procs = len(feature_extractor_procs)
                    for fe_proc_idx in range(num_feature_extractor_procs):
                        p = feature_extractor_procs[fe_proc_idx]
                        if p in killed_extractor_procs:
                            continue
                        if not p.is_alive():
                            logger.info(
                                f"{p.pid} (exitcode: {p.exitcode}) is not alive, but queue still contains tasks!"
                            )
                            logger.info("Joining and starting new process...")
                            p.join()
                            killed_extractor_procs.append(p)
                            p_new = multiprocessing.Process(
                                target=feat_extract_process,
                                args=(input_queue, output_queue),
                            )
                            p_new.start()
                            feature_extractor_procs.append(p_new)

                finally:
                    out_cur = datetime.now()
                    if out_log is None or out_cur - out_log > timedelta(seconds=5):
                        start_delta = out_cur - out_start

                        if processed_files_counter > 0:
                            avg_hours = (
                                (
                                    len(process_header_files)
                                    * (
                                        start_delta.total_seconds()
                                        / processed_files_counter
                                    )
                                )
                                / 60
                                / 60
                            )
                        else:
                            avg_hours = float("nan")

                        logger.info(
                            f"Processed {processed_files_counter}/{len(process_header_files)} in {start_delta} (est {avg_hours} hr)"
                        )
                        out_log = out_cur

    out_cur = datetime.now()
    start_delta = out_cur - out_start
    logger.info(
        f"Finished processing {processed_files_counter}/{len(process_header_files)} in {start_delta}"
    )

    # Close the queues
    input_queue.close()
    input_queue.join_thread()
    output_queue.close()
    output_queue.join_thread()

    # print(input_queue.qsize(), output_queue.qsize(), processed_files_counter)

    # load the data
    logger.info(f"Loading record label mapping from '{labels_fp}'")
    mapped_records = {}
    with open(labels_fp, mode="r", newline="\n") as labelfile:
        for line in labelfile.readlines():
            header_file_path, dxs = json.loads(line)
            mapped_records[header_file_path] = dxs

    logger.info(f"Loading features_df from '{features_fp}'")
    features_df = pd.read_csv(
        features_fp, header=0, names=fieldnames, index_col="header_file", nrows=20
    )
    logger.info("Constructing labels array...")
    labels = [mapped_records[row[0]] for row in features_df.itertuples()]

    # logger.info("Dropping 'header_file' column from features_df")
    # features_df.reset_index(drop=True, inplace=True) # is necessary?

    # Load the SNOMED CT code mapping table
    with open("data/snomed_ct_dx_map.json", "r") as f:
        SNOMED_CODE_MAP = json.load(f)

    logger.info("Loading scoring function weights")
    rows, cols, all_weights = load_table(weights_file)
    assert rows == cols, "rows and cols mismatch"
    scored_codes = rows

    for experiment_num in range(experiments_to_run):
        with ElapsedTimer() as timer:
            logger.info(f"Running experiment #{experiment_num}")

            logger.info("Splitting data into training and evaluation split")
            train_features, eval_features, train_labels, eval_labels = train_test_split(
                features_df, labels, test_size=0.15
            )

            logger.info(f"Training dataset shape: {train_features.shape}")
            logger.info(f"Evaluation dataset shape: {eval_features.shape}")

            to_save_data = {
                "train_records": train_features.index.to_list(),
                "eval_records": eval_features.index.to_list(),
            }

            for idx_sc, sc in enumerate(scored_codes):
                _abbrv, dx = SNOMED_CODE_MAP[str(sc)]
                logger.info(f"Training classifier for {dx} (code {sc})...")

                sc, model = _train_label_classifier(
                    sc,
                    idx_sc,
                    all_weights,
                    train_features,
                    train_labels,
                    eval_features,
                    eval_labels,
                    scored_codes,
                    early_stopping_rounds,
                )

                to_save_data[sc] = model

            _display_metrics(logger, eval_features, eval_labels, to_save_data)
            _save_experiment(logger, output_directory, to_save_data)

        logger.info(f"Experiment {experiment_num} took {timer.duration:.2f} seconds")


def _train_label_classifier(
    sc,
    idx_sc,
    all_weights,
    train_features,
    train_labels,
    eval_features,
    eval_labels,
    scored_codes,
    early_stopping_rounds,
):
    label_weights = all_weights[idx_sc]
    train_labels, train_weights = _determine_sample_weights(
        train_labels, scored_codes, label_weights
    )

    eval_labels, eval_weights = _determine_sample_weights(
        eval_labels, scored_codes, label_weights
    )

    # try negative over positive https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
    pos_count = len([e for e in train_labels if e])
    pos_count = max(pos_count, 1)
    scale_pos_weight = (len(train_labels) - pos_count) / pos_count

    model = XGBClassifier(
        booster="gbtree",  # gbtree, dart or gblinear
        verbosity=0,
        # tree_method="gpu_hist",
        sampling_method="gradient_based",
        scale_pos_weight=scale_pos_weight,
    )

    model = model.fit(
        train_features,
        train_labels,
        sample_weight=train_weights,
        eval_set=[(train_features, train_labels), (eval_features, eval_labels)],
        sample_weight_eval_set=[train_weights, eval_weights],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return sc, model


def _determine_sample_weights(
    data_set, scored_codes, label_weights, weight_threshold=0.5
):
    """Using the scoring labels weights to increase the dataset size of positive labels
    """
    data_labels = []
    sample_weights = []
    for dt in data_set:
        sample_weight = None
        for dx in dt:
            if str(dx) in scored_codes:
                _sample_weight = label_weights[scored_codes.index(str(dx))]
                if _sample_weight < weight_threshold:
                    continue
                if sample_weight is None or _sample_weight > sample_weight:
                    sample_weight = _sample_weight

        if sample_weight is None:
            # not a scored label, treat as a negative example (weight of 1)
            sample_weight = 1.0
            data_labels.append(False)
        else:
            data_labels.append(True)
        sample_weights.append(sample_weight)
    return data_labels, sample_weights


def _display_metrics(logger, features_df, ground_truth, to_save_data):
    classes = []
    labels = []
    scores = []

    for k, v in to_save_data.items():
        if not is_number(k):
            continue

        classes.append(str(k))
        labels.append(v.predict(features_df).tolist())
        scores.append(v.predict_proba(features_df)[:, 1].tolist())

    labels = np.array(labels).T
    scores = np.array(scores).T

    raw_ground_truth_labels = []
    for dx in ground_truth:
        raw_ground_truth_labels.append([
            str(dv) for dv in dx
        ])

    (
        auroc,
        auprc,
        accuracy,
        f_measure,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    ) = evaluate_score_batch(
        predicted_classes=classes,
        predicted_labels=labels,
        predicted_probabilities=scores,
        raw_ground_truth_labels=raw_ground_truth_labels,
    )

    logger.info(
        "AUROC | AUPRC | Accuracy | F-measure | Fbeta-measure | Gbeta-measure | Challenge metric"
    )
    logger.info(
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


def _save_experiment(logger, output_directory, to_save_data):
    logger.info("Saving model...")

    cur_sec = int(time())
    filename = os.path.join(output_directory, f"finalized_model_{cur_sec}.sav")
    joblib.dump(to_save_data, filename, protocol=0)

    logger(f"Saved to {filename}")
