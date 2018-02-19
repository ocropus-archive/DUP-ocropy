# USAGE
# 1. Build new Docker container:
# > docker build -t ocropy -f Dockerfile .
# 2. Run this Docker container:
# > docker run -it --rm -v ${PWD}:/ocropy ocropy bash
# 3. Run tests:
# ># ./run-test

FROM ubuntu:16.04
MAINTAINER Philipp Zumstein
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8

WORKDIR /ocropy
COPY PACKAGES .
RUN apt-get update && \
    apt-get -y install --no-install-recommends git ca-certificates wget unzip && \
    apt-get install -y python-pip $(cat PACKAGES) && \
    git clone --depth 1 'https://github.com/kba/ocr-models-client' /ocr-models-client && \
    /ocr-models-client/ocr-models download -d models 'ocropy/en-default' 'ocropy/fraktur' && \
    pip install --upgrade pip coverage && \
    apt-get -y remove --purge --auto-remove git wget unzip && \
    apt-get clean && rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/*
COPY . .
RUN python setup.py install
