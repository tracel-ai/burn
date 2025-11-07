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

### The ONNX Workflow: Code Generation

The key difference in Burn's approach is **code generation**. Instead of interpreting the ONNX file at runtime, `burn-import` generates native Rust code that represents the model.

#### Conceptual Code Comparison

**PyTorch Model (Original):**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
```

**`burn-import` Generated Rust Code (Conceptual):**
```rust
use burn::prelude::*;
use burn::nn::{Conv2d, Conv2dConfig, ReLU, Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
    conv1: Conv2d<B>,
    relu: ReLU,
    fc1: Linear<B>,
}

impl<B: Backend> MyModel<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        let [batch, c, h, w] = x.dims();
        let x = x.reshape([batch, c * h * w]);
        self.fc1.forward(x)
    }
}
```
This code-generation approach is extremely powerful because the imported model becomes a native Burn `Module`, which can be further modified, trained, or optimized just like any other Burn module.

## Deployment: Taking Your Models Anywhere

One of Burn's main goals is portability. You can train a model on a powerful GPU and then deploy it to a wide range of environments.

### WebAssembly (WASM) for the Browser

The `burn-wgpu` and `burn-ndarray` backends can be compiled to the `wasm32-unknown-unknown` target, allowing you to run your Burn models directly in a web browser.

#### WASM Compilation and Interaction Diagram
```
+--------------------------+     +------------------------+
|   Your Burn Model (Rust) | --> |   Rust Compiler with   |
+--------------------------+     | `wasm32-unknown-unknown`|
                                 |         target         |
                                 +------------------------+
                                           |
                                           V
+--------------------------+     +------------------------+
| `model.wasm` & `model.js`| --> |     Web Application    |
| (Generated by wasm-bindgen)|     | (HTML, CSS, JavaScript)|
+--------------------------+     +------------------------+
         |                                   ^
         `---- (JS calls exported Rust fn) ---'
```
This is perfect for creating interactive web demos, running inference on user data directly in the client's browser (preserving privacy), or building serverless ML applications.

### `no_std` for Embedded Devices

*   **How it works**: The Rust language has a feature that allows you to compile code without linking to the standard library (`std`), which is a requirement for programming on bare-metal or embedded systems with no operating system.
*   **Burn's Support**: The `burn-core` and `burn-ndarray` crates are `no_std` compatible. This means you can define and run your models on microcontrollers and other resource-constrained devices.
*   **Use Cases**: This opens the door to on-device AI for IoT, robotics, and other edge computing applications where you need efficient, low-footprint inference without relying on a network connection.

---

## Exercises

1.  **Fusion Backend**: Modify a previous exercise (e.g., the `MyModel` example from Chapter 5) to use the `Fusion` backend. You will need to wrap your chosen backend (e.g., `Wgpu`) with `Fusion`. While you may not be able to easily measure the performance difference without more complex benchmarking, this exercise will show you how easy it is to enable this feature.
2.  **Explore `burn-import`**: Look at the `SUPPORTED-ONNX-OPS.md` file in the `crates/burn-import` directory. Find two ONNX operators that are supported and two that are marked as "TODO" or are not on the list.
3.  **Conceptual WASM**: If you were to build a web-based application that uses a Burn model for image classification, what would be the main steps involved in getting your Rust-based Burn model to run in the browser? (Hint: Think about the compilation target and how JavaScript and Rust/WASM interact).
4.  **`no_std` Challenge**: Why is the `burn-wgpu` backend *not* `no_std` compatible, while `burn-ndarray` is? (Hint: What does a GPU backend need to do that a CPU backend doesn't, and what operating system or standard library features might that depend on?)
