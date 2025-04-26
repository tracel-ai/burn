# WebAssembly

Burn supports WebAssembly (WASM) execution using the `NdArray` and `WebGpu` backends, allowing
models to run directly in the browser.

Check out the following examples:

- [Image Classification Web](https://github.com/tracel-ai/burn/tree/main/examples/image-classification-web)
- [MNIST Inference on Web](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)

When targeting WebAssembly, certain dependencies require additional configuration. In particular,
the `getrandom` crate requires explicit setting when using `WebGpu`.

Following the [recommended usage](https://github.com/rust-random/getrandom/#webassembly-support),
make sure to explicitly add the dependency with the `wasm_js` feature flag for your project.

```toml
[dependencies]
getrandom = { version = "0.3.2", default-features = false, features = [
    "wasm_js",
] }
```

You also need to set the `getrandom_backend` accordingly via the rust-flags. The flag can either be
set by specifying the `rustflags` field in `.cargo/config.toml`

```toml
[target.wasm32-unknown-unknown]
rustflags = ['--cfg', 'getrandom_backend="wasm_js"']
```

Or by using the `RUSTFLAGS` environment variable:

```
RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
```

This change is now explicitly required with latest versions of Burn, following the `getrandom`
recommendations. This avoids potential issues for WASM developers who do not target Web targets.
