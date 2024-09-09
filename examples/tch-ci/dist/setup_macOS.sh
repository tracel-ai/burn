#! /bin/bash

python3 -m venv pytorch
source pytorch/bin/activate
pip install torch==2.2.0 numpy==1.26.4 setuptools
deactivate