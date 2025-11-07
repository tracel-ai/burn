# Chapter 7: Advanced Features & Deployment

Beyond the core tasks of building and training models, Burn offers a suite of advanced features for performance optimization and deployment. This chapter covers three key areas: kernel fusion for speed, ONNX import for interoperability, and deployment to specialized targets like web browsers (WASM) and embedded devices (`no_std`).

## Performance Tuning with Kernel Fusion

Deep learning models often involve sequences of simple, element-wise operations (e.g., an addition followed by a ReLU activation). On a GPU, each of these operations requires launching a separate "kernel," which has a small but non-trivial overhead. **Kernel fusion** is an optimization technique that merges multiple operations into a single GPU kernel, reducing this overhead and significantly speeding up computation.

Burn implements this via the `burn-fusion` crate, using the same backend decorator pattern we saw with `burn-autodiff`.

### The `Fusion` Backend Decorator

The `Fusion<B>` struct, found in `crates/burn-fusion/src/backend.rs`, is a backend that wraps another backend `B` to add kernel fusion capabilities.

```rust
// crates/burn-fusion/src/backend.rs

#[derive(Clone, Debug, Default)]
pub struct Fusion<B: FusionBackend> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> Backend for Fusion<B> {
    // ...
    type FloatTensorPrimitive = FusionTensor<B::FusionRuntime>;
    // ...
}
```

### How It Works

1.  **Wrapping**: To enable fusion, you wrap your main backend, for example, changing your backend type from `Wgpu` to `Fusion<Wgpu>`.
2.  **Operation Interception**: The `Fusion` backend doesn't execute operations immediately. Instead, its `FusionTensor` primitive intercepts tensor operations and records them in a computational graph.
3.  **Optimization and Fusion**: The `Fusion` backend maintains a client-server architecture. The client (`GlobalFusionClient`) collects the operations. When it's time to execute (e.g., when a tensor's data is actually needed), the client sends the graph of operations to the fusion server. The server has a set of `OptimizationBuilder`s that analyze the graph, find fusible patterns, and merge them into more efficient, combined operations.
4.  **Execution**: The fused operations (and any non-fused ones) are then executed by the inner backend (`Wgpu` in this case).

This lazy-evaluation and fusion system is a powerful, automatic performance enhancement. For many models, simply enabling the `Fusion` backend can lead to significant speedups with no changes to the model code.

## Interoperability: Importing ONNX Models

You don't always have to build your models from scratch in Burn. The `burn-import` crate provides tools to convert models from other popular formats into Burn code. The most prominent of these is ONNX (Open Neural Network Exchange).

### The ONNX Workflow

1.  **Export**: You start with a pre-trained model in another framework, like PyTorch or TensorFlow. You export this model to the ONNX format.
2.  **Convert**: You use the `burn-import` command-line tool or library to parse the `.onnx` file.
3.  **Code Generation**: The tool doesn't just load the weights; it analyzes the model's architecture (the sequence of operations) and **generates Rust code** that defines the model as a Burn `Module`. This is a crucial distinction from many other frameworks that simply interpret the ONNX graph at runtime. By generating native Rust code, the imported model can be optimized by the Rust compiler and Burn's backend systems just like any other Burn module.
4.  **Load and Run**: The generated Rust file can be included in your project. It will contain a `Record` struct and an `init` function to load the converted weights into your newly defined Burn model, which you can then run on any Burn backend.

This code-generation approach is extremely powerful because the imported model becomes a native Burn `Module`, which can be further modified, trained, or optimized just like any other Burn module.

## Deployment: Taking Your Models Anywhere

One of Burn's main goals is portability. You can train a model on a powerful GPU and then deploy it to a wide range of environments.

### WebAssembly (WASM) for the Browser

*   **How it works**: Both the `burn-wgpu` and `burn-ndarray` backends can be compiled to the `wasm32-unknown-unknown` target. This allows you to run your Burn models directly in a web browser.
*   **Use Cases**: This is perfect for creating interactive web demos, running inference on user data directly in the client's browser (preserving privacy), or building serverless ML applications. The `burn-wgpu` backend can even leverage the browser's WebGPU API for GPU-accelerated inference.

### `no_std` for Embedded Devices

*   **How it works**: The Rust language has a feature that allows you to compile code without linking to the standard library (`std`), which is a requirement for programming on bare-metal or embedded systems with no operating system.
*   **Burn's Support**: The `burn-core` and `burn-ndarray` crates are `no_std` compatible. This means you can define and run your models on microcontrollers and other resource-constrained devices.
*   **Use Cases**: This opens the door to on-device AI for IoT, robotics, and other edge computing applications where you need efficient, low-footprint inference without relying on a network connection.

---

## Exercises

1.  **Fusion Backend**: Modify a previous exercise (e.g., the `MyModel` example from Chapter 5) to use the `Fusion` backend. You will need to wrap your chosen backend (e.g., `Wgpu`) with `Fusion`. While you may not be able to easily measure the performance difference without more complex benchmarking, this exercise will show you how easy it is to enable this feature.
2.  **Explore `burn-import`**: Look at the `SUPPORTED-ONNX-OPS.md` file in the `crates/burn-import` directory. Find two ONNX operators that are supported and two that are marked as "TODO" or are not on the list.
3.  **Conceptual WASM**: If you were to build a web-based application that uses a Burn model for image classification, what would be the main steps involved in getting your Rust-based Burn model to run in the browser? (Hint: Think about the compilation target and how JavaScript and Rust/WASM interact).
