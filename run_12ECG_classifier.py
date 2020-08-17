#!/usr/bin/env python
import functools
import os

import joblib

from util.evaluate_12ECG_score import is_number
from util.raw_to_wfdb import convert_to_wfdb_record
from neurokit2_parallel import wfdb_record_to_feature_dataframe


def run_12ECG_classifier(data, header_data, loaded_model):
    # Use your classifier here to obtain a label and score for each class.
    r = convert_to_wfdb_record(data, header_data)
    record_features, _ = wfdb_record_to_feature_dataframe(r)

    featured_classifier_run = functools.partial(
        _partial_run_classifier, record_features=record_features
    )

    # parallel
    # output = joblib.Parallel(n_jobs=-1, verbose=0)(
    #     joblib.delayed(featured_classifier_run)(kv)
    #     for kv in loaded_model.items()
    #     if is_number(kv[0])
    # )

    # map
    output = map(
        featured_classifier_run, [kv for kv in loaded_model.items() if is_number(kv[0])]
    )

    labels, scores, classes = zip(*output)

    # force labels to be integers
    labels = list(map(int, labels))

    # return current_label, current_score, classes
    return labels, scores, classes


def _partial_run_classifier(kv, record_features=None):
    class_val, model = kv
    label = model.predict(record_features)[0]
    score = model.predict_proba(record_features)[0][1]
    return label, score, str(class_val)


def load_12ECG_model(input_directory):
    # load the most recent model from disk
    model_fps = tuple(sorted(os.listdir(input_directory)))

    print(f"Loading {model_fps[-1]} from {input_directory}...")

    filename = os.path.join(input_directory, model_fps[-1])

    loaded_model = joblib.load(filename)

    return loaded_model
