#!/bin/bash

set -eux

pipx install poetry
poetry completions bash >> ~/.bash_completion
poetry install --remove-untracked

source ~/.bashrc

poetry shell
pre-commit install --hook-type commit-msg --hook-type pre-push
eval "$(register-python-argcomplete cz)"
