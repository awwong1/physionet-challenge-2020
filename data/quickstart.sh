#!/usr/bin/env bash

# Download the ECG record archive, if it does not exist
[ -f PhysioNetChallenge2020_Training_CPSC.tar.gz ] && echo "PhysioNetChallenge2020_Training_CPSC.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_CPSC.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_CPSC.tar.gz/

[ -f PhysioNetChallenge2020_Training_2.tar.gz ] && echo "PhysioNetChallenge2020_Training_2.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_2.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_2.tar.gz/

[ -f PhysioNetChallenge2020_Training_StPetersburg.tar.gz ] && echo "PhysioNetChallenge2020_Training_StPetersburg.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_StPetersburg.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_StPetersburg.tar.gz/

[ -f PhysioNetChallenge2020_Training_PTB.tar.gz ] && echo "PhysioNetChallenge2020_Training_PTB.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_PTB.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_PTB.tar.gz/

[ -f PhysioNetChallenge2020_Training_PTB-XL.tar.gz ] && echo "PhysioNetChallenge2020_Training_PTB-XL.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_PTB-XL.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_PTB-XL.tar.gz/

[ -f PhysioNetChallenge2020_Training_E.tar.gz ] && echo "PhysioNetChallenge2020_Training_E.tar.gz exists" || wget -O PhysioNetChallenge2020_Training_E.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_E.tar.gz/

# Verify the archive integrity matches
md5sum --check MD5SUMS

if [ $? -eq 0 ]
then
    for file in PhysioNetChallenge2020_*.tar.gz
    do
        tar -xzvf $file
    done
else
    echo "MD5SUM data integrity failed, abort extraction"
fi
