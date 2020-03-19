import os
import re
from glob import glob

import numpy as np

from . import BaseAgent


class ScikitLearnResultsAgent(BaseAgent):
    HEAD = "| Lead Classifier | Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |"
    SEPR = "|-----------------|---------|----------|-----------|--------|--------|-------|-------|"
    REGEX = "\| (?P<lead_classifier>\w+) \| (?P<dataset>[\w\-\/]+) \| (?P<accuracy>[\d\.]+) \| (?P<f_measure>[\d\.]+) \| (?P<f_beta>[\d\.]+) \| (?P<g_beta>[\d\.]+) \| (?P<auroc>[\d\.]+) \| (?P<auprc>[\d\.]+) \|"
    RES = "| Lead Classifier | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |"
    SEP2 = "|-----------------|----------|-----------|--------|--------|-------|-------|"

    def __init__(self, config):
        super(ScikitLearnResultsAgent, self).__init__(config)
        self.score_fps = sorted(glob(config["input_glob"]))
        self.out_dir = config["out_dir"]

    def run(self):
        outputs = [self.HEAD, self.SEPR]
        for score_fp in self.score_fps:
            with open(score_fp, "r") as f:
                outputs.extend(
                    [
                        fl.strip()
                        for fl in f.readlines()
                        if (fl.strip() not in (self.HEAD, self.SEPR))
                        and ("/Validation" in fl)
                    ]
                )

        means = {}
        for l in outputs:
            self.logger.info(l)
            s = re.search(self.REGEX, l)
            if not s:
                continue
            lead_classifier = s.group("lead_classifier")

            vals = means.get(lead_classifier, [])

            # dataset = s.group("dataset")
            accuracy = float(s.group("accuracy"))
            f_measure = float(s.group("f_measure"))
            f_beta = float(s.group("f_beta"))
            g_beta = float(s.group("g_beta"))
            auroc = float(s.group("auroc"))
            auprc = float(s.group("auprc"))

            data = np.array((accuracy, f_measure, f_beta, g_beta, auroc, auprc))

            vals.append(data)
            means[lead_classifier] = vals

        outputs.append("\n")
        self.logger.info("Determining averages")
        outputs.append(self.RES)
        self.logger.info(self.RES)
        outputs.append(self.SEP2)
        self.logger.info(self.SEP2)

        for lead_classifier, values in means.items():
            mean_values = np.mean(np.stack(values), axis=0)
            line = f"| {lead_classifier} | " + ( " | ".join([f"{v:.4f}" for v in mean_values.tolist()]) ) + " |"
            self.logger.info(line)
            outputs.append(line)

        fp = os.path.join(self.out_dir, "table.md")
        with open(fp, "w") as f:
            f.writelines([f"{output}\n" for output in outputs])
