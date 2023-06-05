FROM rust:1.70-alpine3.18

RUN apk add --no-cache \
    # To get source code
    git \
    # To get musl header files
    musl-dev 

RUN rustup toolchain install nightly \
    && rustup component add clippy rustfmt --toolchain nightly-x86_64-unknown-linux-musl \
    && rustup default nightly

WORKDIR /workspace

RUN git clone https://github.com/burn-rs/burn.git

RUN apk del git