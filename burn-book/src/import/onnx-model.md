# Importing ONNX Models in Burn

## Table of Contents

1. [Introduction](#introduction)
2. [Why Import Models?](#why-import-models)
3. [Understanding ONNX](#understanding-onnx)
4. [Burn's ONNX Support](#burns-onnx-support)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Advanced Configuration](#advanced-configuration)
7. [Loading and Using Models](#loading-and-using-models)
8. [Troubleshooting](#troubleshooting)
9. [Examples and Resources](#examples-and-resources)
10. [Conclusion](#conclusion)

## Introduction

As the field of deep learning continues to evolve, the need for interoperability between different
frameworks becomes increasingly important. Burn, a modern deep learning framework in Rust,
recognizes this need and provides robust support for importing models from other popular frameworks.
This section focuses on importing
[ONNX (Open Neural Network Exchange)](https://onnx.ai/onnx/intro/index.html) models into Burn,
enabling you to leverage pre-trained models and seamlessly integrate them into your Rust-based deep
learning projects.

## Why Import Models?

Importing pre-trained models offers several advantages:

1. **Time-saving**: Avoid the need to train models from scratch, which can be time-consuming and
   resource-intensive.
2. **Access to state-of-the-art architectures**: Utilize cutting-edge models developed by
   researchers and industry leaders.
3. **Transfer learning**: Fine-tune imported models for your specific tasks, benefiting from
   knowledge transfer.
4. **Consistency across frameworks**: Ensure consistent performance when moving from one framework
   to another.

## Understanding ONNX

ONNX (Open Neural Network Exchange) is an open format designed to represent machine learning models.
Key features include:

- **Framework agnostic**: ONNX provides a common format that works across various deep learning
  frameworks.
- **Comprehensive representation**: It captures both the model architecture and trained weights.
- **Wide support**: Many popular frameworks like PyTorch, TensorFlow, and scikit-learn support ONNX
  export.

By using ONNX, you can easily move models between different frameworks and deployment environments.

## Burn's ONNX Support

Burn takes a unique approach to ONNX import, offering several advantages:

1. **Native Rust code generation**: ONNX models are translated into Rust source code, allowing for
   deep integration with Burn's ecosystem.
2. **Compile-time optimization**: The generated Rust code can be optimized by the Rust compiler,
   potentially improving performance.
3. **No runtime dependency**: Unlike some solutions that require an ONNX runtime, Burn's approach
   eliminates this dependency.
4. **Trainability**: Imported models can be further trained or fine-tuned using Burn.
5. **Portability**: The generated Rust code can be compiled for various targets, including
   WebAssembly and embedded devices.
6. **Any Burn Backend**: The imported models can be used with any of Burn's backends.

## Step-by-Step Guide

Let's walk through the process of importing an ONNX model into a Burn project:

### Step 1: Update `build.rs`

First, add the `burn-import` crate to your `Cargo.toml`:

```toml
[build-dependencies]
burn-import = "~0.17"
```

Then, in your `build.rs` file:

```rust
use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

This script uses `ModelGen` to generate Rust code from your ONNX model during the build process.

### Step 2: Modify `mod.rs`

In your `src/model/mod.rs` file, include the generated code:

```rust
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

This makes the generated model code available in your project.

### Step 3: Use the Imported Model

Now you can use the imported model in your Rust code:

```rust
use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::my_model::Model;

fn main() {
    let device = NdArrayDevice::default();

    // Create model instance and load weights from target dir default device.
    // (see more load options below in "Loading and Using Models" section)
    let model: Model<NdArray<f32>> = Model::default();

    // Create input tensor (replace with your actual input)
    let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 3, 224, 224], &device);

    // Perform inference
    let output = model.forward(input);

    println!("Model output: {:?}", output);
}
```

## Advanced Configuration

The `ModelGen` struct offers several configuration options:

```rust
ModelGen::new()
    .input("path/to/model.onnx")
    .out_dir("model/")
    .record_type(RecordType::NamedMpk)
    .half_precision(false)
    .embed_states(false)
    .run_from_script();
```

- `record_type`: Specifies the format for storing weights (Bincode, NamedMpk, NamedMpkGz, or
  PrettyJson).
- `half_precision`: Use half-precision (f16) for weights to reduce model size.
- `embed_states`: Embed model weights directly in the generated Rust code. Note: This requires
  record type `Bincode`.

## Loading and Using Models

Depending on your configuration, you can load models in different ways:

```rust
// Create a new model instance with device. Initializes weights randomly and lazily.
// You can load weights via `load_record` afterwards.
let model = Model::<Backend>::new(&device);

// Load from a file (must specify weights file in the target output directory or copy it from there).
// File type should match the record type specified in `ModelGen`.
let model = Model::<Backend>::from_file("path/to/weights", &device);

// Load from embedded weights (if embed_states was true)
let model = Model::<Backend>::from_embedded(&device);

// Load from the out director location and load to default device (useful for testing)
let model = Model::<Backend>::default();
```

## Troubleshooting

Here are some common issues and their solutions:

1. **Unsupported ONNX operator**: If you encounter an error about an unsupported operator, check the
   [list of supported ONNX operators](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).
   You may need to simplify your model or wait for support to be added.

2. **Build errors**: Ensure that your `burn-import` version matches your Burn version. Also, check
   that the ONNX file path in `build.rs` is correct.

3. **Runtime errors**: If you get errors when running your model, double-check that your input
   tensors match the expected shape and data type of your model.

4. **Performance issues**: If your imported model is slower than expected, try using the
   `half_precision` option to reduce memory usage, or experiment with different `record_type`
   options.

5. **Artifact Files**: You can view the generated Rust code and weights files in the `OUT_DIR`
   directory specified in `build.rs` (usually `target/debug/build/<project>/out`).

## Examples and Resources

For more detailed examples, check out:

1. [MNIST Inference Example](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference)
2. [SqueezeNet Image Classification](https://github.com/tracel-ai/models/tree/main/squeezenet-burn)

These examples demonstrate real-world usage of ONNX import in Burn projects.

## Conclusion

Importing ONNX models into Burn opens up a world of possibilities, allowing you to leverage
pre-trained models from other frameworks while taking advantage of Burn's performance and Rust's
safety features. By following this guide, you should be able to seamlessly integrate ONNX models
into your Burn projects, whether for inference, fine-tuning, or as a starting point for further
development.

Remember that the `burn-import` crate is actively developed, with ongoing work to support more ONNX
operators and improve performance. Stay tuned to the Burn repository for updates and new features!

---

> 🚨**Note**: The `burn-import` crate is in active development. For the most up-to-date information
> on supported ONNX operators, please refer to the
> [official documentation](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).
