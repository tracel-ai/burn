## Getting Started

Your `wasm-bindgen-cli` version must *exactly* match the `wasm-bindgen` version in [Cargo.toml](../../Cargo.toml) since `wasm-bindgen-cli` is implicitly used by `wasm-pack`.

For example, run `cargo install --version 0.2.88 wasm-bindgen-cli --force`. The version in this example command is not guaranteed to be up to date!

Install [PNPM](https://pnpm.io/).

Then in separate terminals:

1. `cd train && dev.sh`
2. `cd web && pnpm i && pnpm dev`

Any changes to `/train` or `burn` should trigger a recompilation. When a new binary is generated, `web` will automatically refresh the page.
