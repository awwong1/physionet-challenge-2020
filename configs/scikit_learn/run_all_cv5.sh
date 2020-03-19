#!/usr/bin/env bash

# Current pwd should be at the main.py level.

# sklearn.tree.DecisionTreeClassifier
# sklearn.tree.ExtraTreeClassifier
# sklearn.ensemble.ExtraTreesClassifier
# sklearn.neighbors.KNeighborsClassifier
# sklearn.neural_network.MLPClassifier
# sklearn.neighbors.RadiusNeighborsClassifier
# sklearn.ensemble.RandomForestClassifier
# sklearn.linear_model.RidgeClassifierCV

classifiers=(
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.neighbors.KNeighborsClassifier
    sklearn.neural_network.MLPClassifier
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
    sklearn.linear_model.RidgeClassifierCV
)

for classifier in ${classifiers[*]}
do
    # https://stackoverflow.com/a/3162500/1942263
    classifier_name="${classifier##*.}"
    for i in {0..4}
    do
        python3 main.py configs/scikit_learn/base.json \
            --override "{\
                \"exp_name\": \"PhysioNet2020/Scikit_Learn/${classifier_name}/cv-${i}\", \
                \"cross_validation\": {\"val_offset\": ${i}}, \
                \"lead_classifier\": {\"name\": \"${classifier}\"} \
            }"
    done
done

# Classifiers Supporting Multilabel Classification
# configs/scikit_learn/decision_tree/run_cv5.sh
# configs/scikit_learn/extra_tree/run_cv5.sh
# configs/scikit_learn/extra_trees/run_cv5.sh
# configs/scikit_learn/k_neighbors/run_cv5.sh
# configs/scikit_learn/mlp/run_cv5.sh
# configs/scikit_learn/radius_neighbors/run_cv5.sh
# configs/scikit_learn/random_forest/run_cv5.sh
# configs/scikit_learn/ridge_classifier_cv/run_cv5.sh
