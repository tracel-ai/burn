# WebAssembly

Burn supports WebAssembly (WASM) execution using the `NdArray` and `WebGpu` backends, allowing
models to run directly in the browser.

Check out the following examples:

- [Image Classification Web](https://github.com/tracel-ai/burn-onnx/tree/main/examples/image-classification-web)
- [MNIST Inference on Web](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)

When targeting WebAssembly, certain dependencies require additional configuration. In particular,
the `getrandom` crate requires explicit setting when using `WebGpu`.
