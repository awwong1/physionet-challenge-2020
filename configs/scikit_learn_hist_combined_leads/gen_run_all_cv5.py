#!/usr/bin/env python3
# generate a shell script for running all of these classifiers

import itertools
import os
import json

CLASSIFIERS = (
    {"name": "sklearn.ensemble.HistGradientBoostingClassifier"},
)

def gen_experiments(cls_idx, cls_config, val_offset):
    cmds = []
    base_cmd = [
        "python3",
        "main.py",
        "configs/scikit_learn_hist_combined_leads/base.json",
        "--override",
    ]
    cls_name = cls_config["name"].split(".")[-1]
    exp_name = f"PhysioNet2020/ScikitLearnHistSimultaneous_CustomMultioutput/{cls_idx:02}-{cls_name}/cv5-{val_offset}"

    override = {
        "exp_name": exp_name,
        "classifier": cls_config,
        "cross_validation": {"fold": 5, "val_offset": val_offset},
        # "variance_threshold": 0.85
        # "select_from_model": {"name": "sklearn.ensemble.RandomForestClassifier"},
        "classifier_chain_order": [3, 8, 7, 6, 5, 4, 0, 1, 2],
        "convert_nans": False
    }

    # no pipeline augmentations
    raw_override = json.dumps(override)
    cmds.append(" ".join(base_cmd + [json.dumps(raw_override),]))

    return cmds


def main():
    script_out = [
        "#!/usr/bin/env bash",
        "# Do not modify by hand. This script is generated!",
        "# Current pwd should be at the main.py level.",
    ]
    for cls_idx, cls_config in enumerate(CLASSIFIERS):
        for val_offset in range(5):
            cmds = gen_experiments(cls_idx, cls_config, val_offset)
            script_out += cmds

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fil_path = os.path.join(dir_path, "run_all_cv5.sh")
    with open(fil_path, "w") as f:
        f.writelines([f"{l}\n" for l in script_out])
    os.chmod(fil_path, 0o744)


if __name__ == "__main__":
    main()
