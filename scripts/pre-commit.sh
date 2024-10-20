#!/bin/bash

set -eux

pre-commit install --hook-type commit-msg --hook-type pre-push
eval "$(register-python-argcomplete cz)"
