FROM python:3.7.6-slim-buster

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER alex.wong@ualberta.ca
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## wget is required to fetch the model checkpoint
## tar is required to extract a tarball archive
RUN apt-get update && apt-get install -y wget tar

# SimpleCNN project setup
# RUN mkdir -p /physionet/experiments/PhysioNet2020/SimpleCNN/checkpoints
# RUN wget -O /physionet/experiments/PhysioNet2020/SimpleCNN/checkpoints/model_best.pth.tar \
#     https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/physionet_2020/experiments/PhysioNet2020/SimpleCNN/checkpoints/model_best.pth.tar
# COPY simplecnn_run_12ECG_classifier.py run_12ECG_classifier.py

# Feature Extraction + Scikit Learn project setup
RUN mkdir -p experiments/PhysioNet2020/Scikit_Learn/
RUN wget -O experiments/PhysioNet2020/Scikit_Learn/ecgpuwave-GradientBoostingClassifier.tar.gz \
    https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/physionet_2020/experiments/PhysioNet2020/Scikit_Learn/ecgpuwave-GradientBoostingClassifier.tar.gz
RUN tar -xvf experiments/PhysioNet2020/Scikit_Learn/ecgpuwave-GradientBoostingClassifier.tar.gz \
    -C experiments/PhysioNet2020/Scikit_Learn/
COPY scikit_fe_run_12ECG_classifier.py run_12ECG_classifier.py

# ecgpuwave build and installation
RUN apt-get update && apt-get install -y gfortran gcc libcurl4-openssl-dev
WORKDIR /physionet/wfdb-10.6.2
RUN ./configure
RUN make install
WORKDIR /physionet/ecgpuwave-1.3.4
RUN make install
WORKDIR /physionet

# Install necessary requirements for running the script
RUN pip install -r requirements.txt
