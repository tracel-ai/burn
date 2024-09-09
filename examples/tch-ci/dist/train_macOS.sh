#! /bin/bash

export LIBTORCH_USE_PYTORCH=1 \
    DYLD_LIBRARY_PATH="$(find pytorch -type d -name "lib" | grep /torch):$DYLD_LIBRARY_PATH" && \
    ./train
