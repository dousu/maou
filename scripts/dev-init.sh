#!/bin/bash

set -eux

pipx install poetry
poetry completions bash >> ~/.bash_completion
poetry sync -E cpu

# shellcheck source=/dev/null
source ~/.bashrc

poetry env activate

poetry cache clear --all -q .
