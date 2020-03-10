#!/usr/bin/env bash

# Current pwd should be at the main.py level.
# Remove the Apex FusedAdam optimizer line 
# if Apex with C extensions is not installed.

for i in {0..4}
do
    python3 main.py configs/sig_fft_lstm/sig_fft_lstm_cv5.json \
        --override "{\
            \"exp_name\": \"PhysioNet2020/SignalFourierTransformLSTM/cv5${i}\", \
            \"cross_validation\": {\"val_offset\": ${i}}, \
            \"optimizer\": {\"name\": \"apex.optimizers.FusedAdam\"} \
        }"
done

# Evaluate on the Training_WFDB folder after training
# for i in {0..4}
# do
#     python3 simplecnn_run_12ECG_classifier.py \
#         experiments/PhysioNet2020/SignalFourierTransformLSTM/cv5-${i}/checkpoints/model_best.pth.tar \
#         --input Training_WFDB \
#         --output out

#     python3 evaluation-2020/evaluate_12ECG_score.py Training_WFDB out SimpleCNN.cv5-${i}.out
# done

# for i in {0..4}
# do
#     echo "SimpleCNN/cv5-${i} results:"
#     cat SimpleCNN.cv5-${i}.out
#     echo
# done
