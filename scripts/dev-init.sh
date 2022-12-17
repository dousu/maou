#!/bin/bash

set -eux

pipx install poetry
poetry completions bash >> ~/.bash_completion
poetry install --remove-untracked

source ~/.bashrc

poetry shell
