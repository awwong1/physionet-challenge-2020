#!/usr/bin/env bash

ROOT=https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/physionet_2020/
# ROOT = https://cloudypipeline.com:9555/api/download/physionet2020training/ # does not match their official checksums

# Download the ECG record archive, if it does not exist
[ -f PhysioNetChallenge2020_Training_CPSC.tar.gz ] && echo "PhysioNetChallenge2020_Training_CPSC.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_CPSC.tar.gz \
"${ROOT}PhysioNetChallenge2020_Training_CPSC.tar.gz"

[ -f PhysioNetChallenge2020_Training_2.tar.gz ] && echo "PhysioNetChallenge2020_Training_2.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_2.tar.gz \
"${ROOT}PhysioNetChallenge2020_Training_2.tar.gz"

[ -f PhysioNetChallenge2020_Training_StPetersburg.tar.gz ] && echo "PhysioNetChallenge2020_Training_StPetersburg.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_StPetersburg.tar.gz \
"${ROOT}PhysioNetChallenge2020_Training_StPetersburg.tar.gz"

[ -f PhysioNetChallenge2020_Training_PTB.tar.gz ] && echo "PhysioNetChallenge2020_Training_PTB.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_PTB.tar.gz \
"${ROOT}PhysioNetChallenge2020_Training_PTB.tar.gz"

[ -f PhysioNetChallenge2020_Training_PTB-XL.tar.gz ] && echo "PhysioNetChallenge2020_Training_PTB-XL.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_PTB-XL.tar.gz \
"${ROOT}PhysioNetChallenge2020_PTB-XL.tar.gz"

[ -f PhysioNetChallenge2020_Training_E.tar.gz ] && echo "PhysioNetChallenge2020_Training_E.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_E.tar.gz \
"${ROOT}PhysioNetChallenge2020_Training_E.tar.gz"

# Verify the archive integrity matches
md5sum --check MD5SUMS

if [ $? -eq 0 ]; then
    for file in PhysioNetChallenge2020_*.tar.gz
    do
        tar -xzvf $file
    done
else
    echo "MD5SUM data integrity failed, abort extraction"
fi
