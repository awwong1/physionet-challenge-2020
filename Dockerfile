FROM python:3.7.3-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER alex.wong@ualberta.ca
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

RUN apt-get update
RUN apt-get upgrade -y

# Pull the required remote repos
RUN git submodule update --init --recursive
# Install python requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Install updated NeuroKit2 library
RUN pip install --upgrade ./NeuroKit
