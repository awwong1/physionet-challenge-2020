#!/usr/bin/env python3
# generate a shell script for running all of these classifiers

import os
import json


CLASSIFIERS = (
    # Multilabel algorithms
    {"name": "sklearn.tree.DecisionTreeClassifier"},
    {"name": "sklearn.tree.ExtraTreeClassifier"},
    {"name": "sklearn.ensemble.ExtraTreesClassifier"},
    {"name": "sklearn.neighbors.KNeighborsClassifier"},
    {"name": "sklearn.neural_network.MLPClassifier"},
    {"name": "sklearn.neighbors.RadiusNeighborsClassifier"},
    {"name": "sklearn.ensemble.RandomForestClassifier"},
    {"name": "sklearn.linear_model.RidgeClassifierCV"},
    # Multiclass algorithms
    {"name": "sklearn.naive_bayes.BernoulliNB"},
    {"name": "sklearn.naive_bayes.GaussianNB"},
    {"name": "sklearn.semi_supervised.LabelPropagation"},
    {"name": "sklearn.semi_supervised.LabelSpreading"},
    {"name": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"},
    {"name": "sklearn.svm.LinearSVC", "kwargs": {"multi_class": "crammer_singer"}},
    {
        "name": "sklearn.linear_model.LogisticRegression",
        "kwargs": {"multi_class": "multinomial"},
    },
    {
        "name": "sklearn.linear_model.LogisticRegressionCV",
        "kwargs": {"multi_class": "multinomial"},
    },
    {"name": "sklearn.neighbors.NearestCentroid"},
    {"name": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"},
    {"name": "sklearn.linear_model.RidgeClassifier"},
    # Multiclass as One-Vs-One
    {"name": "sklearn.svm.NuSVC"},
    {"name": "sklearn.svm.SVC"},
    {
        "name": "sklearn.gaussian_process.GaussianProcessClassifier",
        "kwargs": {"multi_class": "one_vs_one"},
    },
    # Multiclass as One-Vs-The-Rest
    {"name": "sklearn.ensemble.GradientBoostingClassifier"},
    {
        "name": "sklearn.gaussian_process.GaussianProcessClassifier",
        "kwargs": {"multi_class": "one_vs_rest"},
    },
    {"name": "sklearn.svm.LinearSVC", "kwargs": {"multi_class": "ovr"}},
    {
        "name": "sklearn.linear_model.LogisticRegression",
        "kwargs": {"multi_class": "ovr"},
    },
    {
        "name": "sklearn.linear_model.LogisticRegressionCV",
        "kwargs": {"multi_class": "ovr"},
    },
    {"name": "sklearn.linear_model.SGDClassifier"},
    {"name": "sklearn.linear_model.Perceptron"},
    {"name": "sklearn.linear_model.PassiveAggressiveClassifier"},
)

FEATURE_SELECTION = (
    None,
    # {"name": "sklearn.ensemble.RandomForestClassifier"},
    # {"name": "sklearn.tree.DecisionTreeClassifier"},
)


def run_experiment(feat_selector, cls_idx, cls_config, val_offset):
    cls_name = cls_config["name"].split(".")[-1]
    exp_name = f"PhysioNet2020/ScikitLearn/{cls_idx:02}-{cls_name}/cv5-{val_offset}"
    if feat_selector:
        feat_name = feat_selector["name"].split(".")[-1]
        exp_name = f"PhysioNet2020/ScikitLearn-{feat_name}_Feature_Selection/{cls_idx:02}-{cls_name}/cv5-{val_offset}"
    override = {
        "exp_name": exp_name,
        "lead_classifier": cls_config,
        "cross_validation": {"fold": 5, "val_offset": val_offset},
        "serialize_model": False
    }
    if feat_selector:
        override["lead_pipeline_feature_selection"] = feat_selector

    raw_config_override = json.dumps(override)

    cmd = " ".join((
        "python3",
        "main.py",
        "configs/scikit_learn/base.json",
        "--override",
        json.dumps(raw_config_override),
    ))
    return cmd


def main():
    script_out = [
        "#!/usr/bin/env bash",
        "# Do not modify by hand. This script is generated!",
        "# Current pwd should be at the main.py level."
    ]
    for feat_selector in FEATURE_SELECTION:
        for cls_idx, cls_config in enumerate(CLASSIFIERS):
            for val_offset in range(5):
                cmd = run_experiment(feat_selector, cls_idx, cls_config, val_offset)
                script_out.append(cmd)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fil_path = os.path.join(dir_path, "run_all_cv5.sh")
    with open(fil_path, "w") as f:
        f.writelines([f"{l}\n" for l in script_out])
    os.chmod(fil_path, 0o744)

if __name__ == "__main__":
    main()
