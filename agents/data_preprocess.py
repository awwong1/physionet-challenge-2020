import os
from functools import partial
from glob import glob
from multiprocessing import Pool, freeze_support

import numpy as np
import wfdb
from scipy import signal
from tqdm import tqdm

from util.config import init_class

from . import BaseAgent


class DatasetPreprocessAgent(BaseAgent):
    """Agent to preprocess the PhysioNet 2020 challenge dataset.
    Outputs a fast serialized representation of the underlying *.hea & *.mat files.

    Experiment Configuration Keys:
    - dataset (dict): PhysioNet 2020 Challenge Dataset configuration
    - proc (int, optional): number of workers to use
    - out_dir (str, optional): path to output serialized/processed data
    """

    def __init__(self, config):
        super(DatasetPreprocessAgent, self).__init__(config)
        try:
            proc = len(os.sched_getaffinity(0))
        except Exception:
            proc = 0
        self.out_dir = config.get("out_dir", "data")
        self.proc = config.get("proc", proc)
        self.ds = init_class(config.get("dataset"), proc=self.proc)
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        digits = len(str(len(self.ds)))
        with tqdm(
            range(len(self.ds)), desc="Building processed dataset"
        ) as t:
            if self.proc <= 0:
                out = (DatasetPreprocessAgent.save_file(
                    self.ds[i], ds=self.ds, digits=digits, out_dir=self.out_dir) for i in t)
            else:
                freeze_support()
                with Pool(
                    self.proc,
                    initializer=tqdm.set_lock,
                    initargs=(tqdm.get_lock(),),
                ) as p:
                    out = tuple(p.imap_unordered(
                        partial(DatasetPreprocessAgent.save_file,
                                ds=self.ds, digits=digits, out_dir=self.out_dir),
                        t))
        self.logger.info("Saved %d files to %s", len(out), self.out_dir)

    @staticmethod
    def save_file(idx, ds=None, digits=1, out_dir="data"):
        signal, target = ds[idx]
        file_out = os.path.join(out_dir, f"{idx}".zfill(digits))
        np.savez(file_out, signal=signal, target=target)
        return file_out
