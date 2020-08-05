import csv
import multiprocessing
import os
import queue
import json
from glob import glob
from datetime import datetime, timedelta

import wfdb

from util.log import configure_logging
from neurokit2_parallel import (
    ECG_LEAD_NAMES,
    KEYS_INTERVALRELATED,
    KEYS_TSFRESH,
    wfdb_record_to_feature_dataframe,
)


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
    input_directory, output_directory, labels_fp="dxs.txt", features_fp="features.csv"
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
    header_files = tuple(
        hfp
        for hfp in sorted(
            glob(os.path.join(input_directory, "**/*.hea"), recursive=True)
        )
        if hfp not in mapped_records
    )

    logger.info("Number of ECG records to process: %d", len(header_files))

    num_cpus = len(os.sched_getaffinity(0))
    logger.info("Number of available CPUs: %d", num_cpus)

    # Setup & populate input queue, then initialize output queue
    input_queue = multiprocessing.JoinableQueue()
    for header_file in header_files:
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

    # initialize the header if the file does not exist
    if not os.path.isfile(features_fp):
        with open(features_fp, "w", newline="\n") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=_get_fieldnames())
            writer.writeheader()

    with open(features_fp, "a", newline="\n") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_get_fieldnames())
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
                            logger.info(f"{p.pid} (exitcode: {p.exitcode}) is not alive, but queue still contains tasks!")
                            logger.info("Joining and starting new process...")
                            p.join()
                            killed_extractor_procs.append(p)
                            p_new = multiprocessing.Process(
                                target=feat_extract_process, args=(input_queue, output_queue)
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
                                    len(header_files)
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
                            f"Processed {processed_files_counter}/{len(header_files)} in {start_delta} (est {avg_hours} hr)"
                        )
                        out_log = out_cur

    out_cur = datetime.now()
    start_delta = out_cur - out_start
    logger.info(
        f"Finished processing {processed_files_counter}/{len(header_files)} in {start_delta}"
    )

    # Close the queues
    input_queue.close()
    input_queue.join_thread()
    output_queue.close()
    output_queue.join_thread()

    # print(input_queue.qsize(), output_queue.qsize(), processed_files_counter)

    # TODO: split data, train XGBClassifier, report results
