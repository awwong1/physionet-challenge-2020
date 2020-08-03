import multiprocessing
import os
import queue
from glob import glob

import wfdb
import tsfresh

from util.log import configure_logging
from neurokit2_parallel import (
    get_intervalrelated_features,
    ecg_clean,
    best_heartbeats_from_ecg_signal,
)


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
        cleaned_signals = ecg_clean(r.p_signal, sampling_rate=r.fs)

        # interval related features from parallel neurokit2
        intervalrelated_features, proc_df = get_intervalrelated_features(
            r.p_signal, cleaned_signals, sampling_rate=r.fs, ecg_lead_names=r.sig_name
        )

        output_queue.put(intervalrelated_features)

        # tsfresh related features from heartbeats
        best_heartbeat_df = best_heartbeats_from_ecg_signal(
            proc_df, sampling_rate=r.fs, ecg_lead_names=r.sig_name
        )
        heartbeat_features = tsfresh.extract_features(
            best_heartbeat_df, column_id="lead", column_sort="time"
        )


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
    while True:
        try:
            fp = output_queue.get(True, 0.1)
            output_queue.task_done()
            processed_files_counter += 1
            print(fp)
        except queue.Empty:
            # When the output queue is empty and all workers are terminated
            # all files have been processed
            if all(not p.is_alive() for p in feature_extractor_procs):
                break

    # Close the queues
    input_queue.close()
    input_queue.join_thread()
    output_queue.close()
    output_queue.join_thread()

    print(input_queue.qsize(), output_queue.qsize(), processed_files_counter)
