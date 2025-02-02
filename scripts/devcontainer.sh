#!/bin/bash

set -eux

# 必要ならインストールするが少しでも軽くしたいので基本しない
# sudo apt update
# sudo apt install -y build-essential
# sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# gcloud CLIのインストール
# https://cloud.google.com/sdk/docs/install?hl=ja#linux
cd ~
curl -O "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-$(uname -m).tar.gz"
tar -xf "google-cloud-cli-linux-$(uname -m).tar.gz" --overwrite
./google-cloud-sdk/install.sh --usage-reporting false --path-update true --rc-path ~/.bashrc --bash-completion true
cd -
