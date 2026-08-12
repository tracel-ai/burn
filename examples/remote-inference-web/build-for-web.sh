#!/usr/bin/env bash

# Add wasm32 target for compiler.
rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &> /dev/null
then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
fi

# Iroh's wasm randomness backend is selected through getrandom's wasm_js cfg.
RUSTFLAGS='-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 --cfg web_sys_unstable_apis --cfg getrandom_backend="wasm_js"'

mkdir -p pkg
wasm-pack build --out-dir pkg --release --target web --no-typescript
