
# Add wasm32 target for compiler.
rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &> /dev/null
then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
    exit
fi

# Run wasm pack tool to build JS wrapper files and copy wasm to pkg directory.
wasm-pack build --out-dir pkg --release --target web --no-typescript
