#!/usr/bin/env python
import os

import joblib

from util.evaluate_12ECG_score import is_number
from util.raw_to_wfdb import convert_to_wfdb_record
from neurokit2_parallel import wfdb_record_to_feature_dataframe


def run_12ECG_classifier(data, header_data, loaded_model):
    # Use your classifier here to obtain a label and score for each class.
    r = convert_to_wfdb_record(data, header_data)
    record_features, dx = wfdb_record_to_feature_dataframe(r)

    classes = []
    labels = []
    scores = []
    for k, v in loaded_model.items():
        if not is_number(k):
            continue
        classes.append(str(k))
        labels.append(int(v.predict(record_features)[0]))
        scores.append(v.predict_proba(record_features)[0][1])

    # return current_label, current_score, classes
    return labels, scores, classes


def load_12ECG_model(input_directory):
    # load the most recent model from disk
    model_fps = tuple(sorted(os.listdir(input_directory)))

    print(f"Loading {model_fps[-1]} from {input_directory}...")

    filename = os.path.join(input_directory, model_fps[0])

    loaded_model = joblib.load(filename)

    return loaded_model
