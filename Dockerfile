FROM rust:1.70-slim-bullseye

RUN apt update \
    && apt install -y \
    # To clone our repo
    git \
    # Needs for compiling openssl
    pkg-config \
    libssl-dev \
    # Needs for compiling torch-sys
    g++ \
    # Needs for downloading libtorch
    axel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN axel -qo /opt/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip \
    && cd /opt \
    && unzip -qq libtorch.zip \
    && rm /opt/libtorch.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

RUN rustup component add \
    # Used in `burn-import/src/formater.rs`
    rustfmt \
    clippy
    
WORKDIR /workspace

RUN git clone https://github.com/burn-rs/burn.git

RUN apt autoremove -y \
    git \
    unzip \
    axel