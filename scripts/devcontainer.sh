#!/bin/bash

set -eux

sudo apt update

# scikit-learnのインストールに必要
sudo apt install gfortran libopenblas-dev

# gcloud CLIのインストール
# https://cloud.google.com/sdk/docs/install?hl=ja#linux
cd ~
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-$(uname -m).tar.gz
tar -xf google-cloud-cli-linux-$(uname -m).tar.gz --overwrite
./google-cloud-sdk/install.sh --usage-reporting false --path-update true --rc-path ~/.bashrc --bash-completion true
cd -
