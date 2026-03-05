#!/bin/bash

set -eux

# Cargo registryボリュームの権限修正(rootで作成されるため)
sudo chown -R vscode:rustlang /usr/local/cargo/registry
sudo chmod -R g+ws /usr/local/cargo/registry

# 必要ならインストールするが少しでも軽くしたいので基本しない
# build-essentialとか
sudo apt update
sudo apt install -y cmake tmux clang lld
sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# gcloud CLIのインストール
# https://cloud.google.com/sdk/docs/install?hl=ja#linux
cd ~
curl -O "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-$(uname -m).tar.gz"
tar -xf "google-cloud-cli-linux-$(uname -m).tar.gz" --overwrite
./google-cloud-sdk/install.sh --usage-reporting false --path-update true --rc-path ~/.bashrc --bash-completion true
cd -

# aws CLIのインストール
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
cd ~
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -qo awscliv2.zip
sudo ./aws/install --update
cd -
