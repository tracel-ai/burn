#! /bin/bash

curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu121.zip
unzip libtorch.zip
rm libtorch.zip

mkdir .cargo

cat <<EOF > .cargo/config.toml
[env]
LD_LIBRARY_PATH = "$(pwd)/libtorch/lib:$LD_LIBRARY_PATH"
LIBTORCH = "$(pwd)/libtorch"
EOF
