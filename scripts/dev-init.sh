#!/bin/bash

set -eux

pipx install poetry
poetry completions bash >> ~/.bash_completion
poetry install --sync

# shellcheck source=/dev/null
source ~/.bashrc

poetry shell
