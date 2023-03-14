
# Add wasm32 target for compiler.
rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &> /dev/null
then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
    exit
fi

# Set optimization flags
export RUSTFLAGS="-C lto=fat -C embed-bitcode=yes -C codegen-units=1 -C opt-level=3"

# Run wasm pack tool to build JS wrapper files and copy wasm to pkg directory.
mkdir -p pkg
wasm-pack build --out-dir pkg --release --target web --no-typescript

