#!/usr/bin/env bash

# Current pwd should be at the main.py level.
# Remove the Apex FusedAdam optimizer line 
# if Apex with C extensions is not installed.

for i in {0..4}
do
    python3 main.py configs/simple_cnn/simple_cnn_cv5.json \
        --override "{\
            \"exp_name\": \"PhysioNet2020/SimpleCNN/cv5-${i}\", \
            \"cross_validation\": {\"val_offset\": ${i}}, \
            \"optimizer\": {\"name\": \"apex.optimizers.FusedAdam\"}
        }"
done
