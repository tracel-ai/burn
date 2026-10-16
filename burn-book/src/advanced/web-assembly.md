# WebAssembly

Burn supports WebAssembly (WASM) execution using the `Flex` and `WebGpu` backends, allowing models
to run directly in the browser.

Check out the following examples:

- [Image Classification Web](https://github.com/tracel-ai/burn-onnx/tree/main/examples/image-classification-web)
- [MNIST Inference on Web](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)

When targeting WebAssembly, certain dependencies require additional configuration. In particular,
the `getrandom` crate requires explicit setting when using `WebGpu`.

## WebGPU startup

Burn's `WebGpu` backend uses CubeCL. CubeCL's compiler uses Pliron, which initializes its registries
through `inventory` constructors. With wasm-bindgen, the Wasm linker can run these constructors on
every call from JavaScript, including async callbacks. This repeated work caused the WebGPU slowdown
reported in [Burn #5992](https://github.com/tracel-ai/burn/issues/5992). This upstream
initialization behavior requires an explicit startup call in the application, as shown below.
Upgrading Burn or CubeCL alone does not prevent it.

For `wasm32-unknown-unknown`, call `__wasm_call_ctors()` once at the beginning of your application's
`#[wasm_bindgen(start)]` function. This tells the linker that the application handles
initialization, so it stops adding constructor calls to every exported function. If you already have
a startup function, add the call there:

```rust,ignore
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn start() {
    unsafe extern "C" {
        fn __wasm_call_ctors();
    }

    // SAFETY: called once during module startup, before constructor-dependent work.
    unsafe {
        __wasm_call_ctors();
    }

    // Set up the application or spawn async work here.
}
```

This runs all module constructors, including those from other libraries, so do not call it again.
Keep setup that uses Burn's `WebGpu` backend after this call in the same startup function, since
[separate startup hooks have no guaranteed order](https://wasm-bindgen.github.io/wasm-bindgen/reference/attributes/on-rust-exports/start.html).

See
[inventory's Wasm guidance](https://docs.rs/inventory/latest/inventory/#webassembly-and-constructors)
and the [upstream wasm-bindgen discussion](https://github.com/wasm-bindgen/wasm-bindgen/issues/1216)
for more background.
