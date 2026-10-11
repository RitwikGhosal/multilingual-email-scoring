#!/usr/bin/env bash
set -e

# TorchAO is not required by this project. Some Colab
# environments include an older release that conflicts
# with PEFT's optional TorchAO compatibility check.
python -m pip uninstall -y torchao

python -m pip install -r requirements.txt
