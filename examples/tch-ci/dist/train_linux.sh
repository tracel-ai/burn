#! /bin/bash

export LD_LIBRARY_PATH="$(pwd)/libtorch/lib:$LD_LIBRARY_PATH" \
    LIBTORCH="$(pwd)/libtorch:$LIBTORCH" && \
    ./train
