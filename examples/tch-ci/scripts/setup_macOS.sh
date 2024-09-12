#! /bin/bash

python3 -m venv .venv

source .venv/bin/activate

pip install torch==2.2.0 numpy==1.26.4 setuptools

mkdir .cargo

cat <<EOF > .cargo/config.toml
[env]
LIBTORCH_USE_PYTORCH = "1"
DYLD_LIBRARY_PATH = "$(pwd)/$(find .venv -type d -name "lib" | grep /torch):$DYLD_LIBRARY_PATH"
EOF

deactivate
