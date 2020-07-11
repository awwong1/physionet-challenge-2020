#!/usr/bin/env python
import os

import joblib

from feature_extractor import data_header_to_np_array, structured_np_array_to_features


def run_12ECG_classifier(data, header_data, loaded_model):
    # Use your classifier here to obtain a label and score for each class.
    np_array = data_header_to_np_array(data, header_data)
    features = structured_np_array_to_features(np_array)

    classes = []
    labels = []
    scores = []
    for k, v in loaded_model.items():
        if k in ("train_records", "eval_records"):
            continue
        classes.append(str(k))
        labels.append(int(v.predict(features)[0]))
        scores.append(v.predict_proba(features)[0][1])

    # return current_label, current_score, classes
    return labels, scores, classes


def load_12ECG_model(input_directory):
    # load the most recent model from disk
    model_fps = tuple(sorted(os.listdir(input_directory)))

    print(f"Loading {model_fps[-1]} from {input_directory}...")

    filename = os.path.join(input_directory, model_fps[0])

    loaded_model = joblib.load(filename)

    return loaded_model
