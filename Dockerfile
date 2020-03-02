FROM python:3.7.6-slim-buster

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER alex.wong@ualberta.ca
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.
RUN apt-get update && apt-get install -y wget
RUN wget https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/physionet_2020/experiments/PhysioNet2020/SimpleCNN/checkpoints/model_best.pth.tar

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
