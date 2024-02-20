The `burn-no-std-tests` contains integration tests aimed to check `no_std` compatibility of `burn`, `burn-core`, `burn-tensor` and `burn-ndarray` packages.

Currently there is only a minimal test that checks if mnist model can be built with `no_std`. More tests should be added to check completeness.

The continuous integration (CI) should build with additional targets:

 * `wasm32-unknown-unknown` - WebAssembly
 * `thumbv7m-none-eabi` - ARM Cortex-M3

Shell commands to build and test the package:

```sh

# install the new targets if not installed previously
rustup target add thumbv7m-none-eabi
rustup target add wasm32-unknown-unknown

# build for various targets 
cargo build # regular build
cargo build --target thumbv7m-none-eabi
cargo build --target wasm32-unknown-unknown

# test
cargo test

 ```