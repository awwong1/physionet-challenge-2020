import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from driver import get_classes, load_challenge_data
from util.sigproc import extract_record_features

from . import BaseAgent


class FeatureExtractionAgent(BaseAgent):
    """Agent for converting the ECG signals and data into fixed size tabular input features
    """

    def __init__(self, config):
        super(FeatureExtractionAgent, self).__init__(config)

        self.input_directory = config.get("input_directory", "Training_WFDB")
        self.out_dir = config["out_dir"]
        self.erf_options = config.get("extract_record_features", {})

        input_files = []
        for f in os.listdir(self.input_directory):
            if (
                os.path.isfile(os.path.join(self.input_directory, f))
                and not f.lower().startswith(".")
                and f.lower().endswith("mat")
            ):
                input_files.append(f)

        self.input_files = tuple(sorted(input_files))
        self.tqdm = tqdm(self.input_files, desc=f"Extracting {self.input_directory}")

    def run(self):
        for input_file in self.tqdm:
            fp = os.path.join(self.input_directory, input_file)
            data, headers = load_challenge_data(fp)

            features = extract_record_features(data, headers, **self.erf_options)

            sex = features.pop("sex")
            age = features.pop("age")
            target = features.pop("target")

            sig = features.pop("sig")
            # split out each lead as as its own feature array
            np_savez_data = OrderedDict()
            for sig_name, sig_features in sig.items():
                np_savez_data[sig_name] = np.concatenate([
                    sex.flatten(),
                    age.flatten(),
                    *[v.flatten() for v in sig_features.values()]
                ])

            # assert len(inputs) == 4479, f"Input length not consistent on {input_file}"

            bn = os.path.splitext(input_file)[0]
            out_f = os.path.join(self.out_dir, f"{bn}.npz")

            np.savez(out_f, target=target, **np_savez_data)

    def finalize(self):
        if self.tqdm:
            self.tqdm.close()
