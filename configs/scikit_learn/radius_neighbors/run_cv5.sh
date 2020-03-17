#!/usr/bin/env bash

# Current pwd should be at the main.py level.

for i in {0..4}
do
    python3 main.py configs/scikit_learn/radius_neighbors/radius_neighbors_cv5.json \
        --override "{\
            \"exp_name\": \"PhysioNet2020/Scikit_Learn/RadiusNeighborsClassifier/cv-${i}\", \
            \"cross_validation\": {\"val_offset\": ${i}} \
        }"
done
