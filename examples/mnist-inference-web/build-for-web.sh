#!/usr/bin/env bash

# Add wasm32 target for compiler.
rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &> /dev/null
then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
fi

# Set optimization flags
export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 --cfg web_sys_unstable_apis --cfg getrandom_backend=\"wasm_js\""

# Run wasm pack tool to build JS wrapper files and copy wasm to pkg directory.
mkdir -p pkg
wasm-pack build --out-dir pkg --release --target web --no-typescript --no-default-features --features $1

