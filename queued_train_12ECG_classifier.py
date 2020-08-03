import csv
import multiprocessing
import os
import queue
from glob import glob
from datetime import datetime, timedelta

import wfdb
import tsfresh

from util.log import configure_logging
from neurokit2_parallel import (
    ECG_LEAD_NAMES,
    KEYS_INTERVALRELATED,
    KEYS_TSFRESH,
    get_intervalrelated_features,
    ecg_clean,
    best_heartbeats_from_ecg_signal,
    signal_to_tsfresh_df,
)


def _get_fieldnames():
    field_names = []
    for lead_name in ECG_LEAD_NAMES:
        for key in KEYS_INTERVALRELATED:
            field_names.append(f"{lead_name}_{key}")
        for key in KEYS_TSFRESH:
            hb_key = f"hb_sig__{key}"
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

        r.sig_name = ECG_LEAD_NAMES  # force consistent naming

        cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)

        # interval related features from parallel neurokit2
        intervalrelated_features, proc_df = get_intervalrelated_features(
            r.p_signal, cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
        )

        # tsfresh related features from heartbeats
        best_heartbeat_df = best_heartbeats_from_ecg_signal(
            proc_df, sampling_rate=r.fs, ecg_lead_names=r.sig_name
        )
        heartbeat_features = tsfresh.extract_features(
            best_heartbeat_df,
            column_id="lead",
            column_sort="time",
            n_jobs=0,
            disable_progressbar=True,
        )

        # tsfresh related features from signal (offset 1 second, maximum 2500 samples?)
        signal_df = signal_to_tsfresh_df(
            cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
        )
        full_waveform_features = tsfresh.extract_features(
            signal_df,
            column_id="lead",
            column_sort="time",
            n_jobs=0,
            disable_progressbar=True,
        )

        # flatten and combine three feature vectors into single hashable item
        ecg_features = {}
        for lead_name in r.sig_name:
            irf_groupby = intervalrelated_features.groupby("ECG_Sig_Name")
            irf_group = irf_groupby.get_group(lead_name)
            for key in KEYS_INTERVALRELATED:
                ecg_features[f"{lead_name}_{key}"] = irf_group[key][0]
            for key in KEYS_TSFRESH:
                hb_key = f"hb_sig__{key}"
                ecg_features[f"{lead_name}_{hb_key}"] = heartbeat_features.loc[
                    lead_name
                ][hb_key]
            for key in KEYS_TSFRESH:
                sig_key = f"sig__{key}"
                ecg_features[f"{lead_name}_{sig_key}"] = full_waveform_features.loc[
                    lead_name
                ][sig_key]

        output_queue.put(ecg_features)


def train_12ECG_classifier(input_directory, output_directory):
    logger = configure_logging()

    logger.info("Finding input files...")
    header_files = tuple(
        glob(os.path.join(input_directory, "**/*.hea"), recursive=True)
    )

    logger.info("Number of ECG records: %d", len(header_files))

    num_cpus = len(os.sched_getaffinity(0))
    logger.info("Number of available CPUs: %d", num_cpus)

    # Setup & populate input queue, then initialize output queue
    input_queue = multiprocessing.JoinableQueue()
    for header_file in header_files:
        input_queue.put_nowait(header_file)
    output_queue = multiprocessing.JoinableQueue()

    # all CPUs except 1 used for feature extraction
    num_feature_extractor_procs = max(num_cpus - 1, 1)
    feature_extractor_procs = []
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

    with open("features.csv", "w", newline="\n") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_get_fieldnames())
        writer.writeheader()

        while True:
            try:
                f_dict = output_queue.get(True, 0.1)
                writer.writerow(f_dict)
                output_queue.task_done()
                processed_files_counter += 1
            except queue.Empty:
                # When the output queue is empty and all workers are terminated
                # all files have been processed
                if all(not p.is_alive() for p in feature_extractor_procs):
                    break
            finally:
                out_cur = datetime.now()
                if out_log is None or out_cur - out_log > timedelta(seconds=5):
                    start_delta = out_cur - out_start
                    logger.info(
                        f"Processed {processed_files_counter}/{len(header_files)} in {start_delta}"
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
