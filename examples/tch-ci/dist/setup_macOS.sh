#! /bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.0 numpy==1.26.4 setuptools
deactivate
