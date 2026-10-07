# WebAssembly

Burn supports WebAssembly (WASM) execution using the `Flex` and `WebGpu` backends, allowing
models to run directly in the browser.

Check out the following examples:

- [Image Classification Web](https://github.com/tracel-ai/burn-onnx/tree/main/examples/image-classification-web)
- [MNIST Inference on Web](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)

When targeting WebAssembly, certain dependencies require additional configuration. In particular,
the `getrandom` crate requires explicit setting when using `WebGpu`.

### Run static constructors once

Burn's GPU backends depend on crates that register static constructors. When a
`wasm32-unknown-unknown` module never calls `__wasm_call_ctors` itself, the linker inserts a call
to it at the start of **every exported function**, so each call from JavaScript (including
wasm-bindgen's internal exports used by async callbacks) re-runs all of them. This can make
inference several times slower. Call it once from your start function:

```rust, ignore
#[cfg(target_family = "wasm")]
unsafe extern "C" {
    fn __wasm_call_ctors();
}

#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(target_family = "wasm")]
    // SAFETY: called once, before any other export.
    unsafe {
        __wasm_call_ctors();
    }
}
```
