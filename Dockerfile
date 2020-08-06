# FROM python:3.7.3-stretch
FROM nvidia/cuda:10.1-cudnn7-devel

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER alex.wong@ualberta.ca
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install git python3 python3-pip -y

# Pull the required remote repos
RUN git submodule update --init --recursive
# Install python requirements
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
# Install updated NeuroKit2 library
RUN python3 -m pip install --upgrade ./NeuroKit
