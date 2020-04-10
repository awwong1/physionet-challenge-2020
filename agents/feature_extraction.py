import os
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from driver import get_classes, load_challenge_data
from util.sigdat import convert_to_wfdb_record, extract_features

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
        self.tqdm = tqdm(self.input_files[1:], desc=f"Extracting {self.input_directory}")

    @staticmethod
    def proc_extract_features(input_file, input_dir="", output_dir=""):
        try:
            fp = os.path.join(input_dir, input_file)
            data, headers = load_challenge_data(fp)
            r = convert_to_wfdb_record(data, headers)

            features = extract_features(r, ann_dir=output_dir)

            sex = features.pop("sex")
            age = features.pop("age")
            target = features.pop("target")

            sig = features.pop("sig")
            # split out each lead as as its own feature array
            np_savez_data = OrderedDict()
            for sig_name, sig_features in sig.items():
                np_savez_data[sig_name] = np.concatenate(
                    [
                        sex.flatten(),
                        age.flatten(),
                        *[v.flatten() for v in sig_features.values()],
                    ]
                )

            bn = os.path.splitext(input_file)[0]
            out_f = os.path.join(output_dir, f"{bn}.npz")

            np.savez(out_f, target=target, **np_savez_data)
            meta_payload = features.pop("meta", {})
            return input_file, meta_payload
        except Exception as e:
            print("ERROR:", input_file)
            raise e

    def run(self):
        worker_fn = partial(
            FeatureExtractionAgent.proc_extract_features,
            input_dir=self.input_directory,
            output_dir=self.out_dir,
        )

        with Pool(
            len(os.sched_getaffinity(0)),
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),),
        ) as p:
            meta = dict(p.imap_unordered(worker_fn, self.tqdm,))

        # single process
        # meta = dict(worker_fn(rn) for rn in self.tqdm)

        # do some logging for weirdness in the dataset
        for input_file, meta_payload in meta.items():
            unsupported_symbols = set().union(*meta_payload["unsupported_symbols"].values())
            if unsupported_symbols:
                self.logger.info(f"{input_file} unsupported symbols {unsupported_symbols}")

    def finalize(self):
        if self.tqdm:
            self.tqdm.close()
