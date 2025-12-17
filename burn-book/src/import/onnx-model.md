# Importing ONNX Models in Burn

## Introduction

As deep learning evolves, interoperability between frameworks becomes crucial. Burn, a modern deep
learning framework in Rust, provides robust support for importing models from other popular
frameworks. This section focuses on importing
[ONNX (Open Neural Network Exchange)](https://onnx.ai/onnx/intro/index.html) models into Burn,
enabling you to leverage pre-trained models in your Rust-based deep learning projects.

## Why Import Models?

Importing pre-trained models offers several advantages:

1. **Time-saving**: Skip the resource-intensive process of training models from scratch.
2. **Access to state-of-the-art architectures**: Utilize cutting-edge models developed by
   researchers and industry leaders.
3. **Transfer learning**: Fine-tune imported models for your specific tasks, benefiting from
   knowledge transfer.
4. **Consistency across frameworks**: Maintain consistent performance when moving between
   frameworks.

## Understanding ONNX

ONNX (Open Neural Network Exchange) is an open format designed to represent machine learning models
with these key features:

- **Framework agnostic**: Provides a common format that works across various deep learning
  frameworks.
- **Comprehensive representation**: Captures both the model architecture and trained weights.
- **Wide support**: Compatible with popular frameworks like PyTorch, TensorFlow, and scikit-learn.

This standardization allows seamless movement of models between different frameworks and deployment
environments.

## Burn's ONNX Support

Burn's approach to ONNX import offers unique advantages:

1. **Native Rust code generation**: Translates ONNX models into Rust source code for deep
   integration with Burn's ecosystem.
2. **Compile-time optimization**: Leverages the Rust compiler to optimize the generated code,
   potentially improving performance.
3. **No runtime dependency**: Eliminates the need for an ONNX runtime, unlike many other solutions.
4. **Trainability**: Allows imported models to be further trained or fine-tuned using Burn.
5. **Portability**: Enables compilation for various targets, including WebAssembly and embedded
   devices.
6. **Backend flexibility**: Works with any of Burn's supported backends.

## ONNX Compatibility

Burn recommends ONNX models use **opset version 16 or higher** for best compatibility. While models
with older opset versions may work, opset 16+ ensures access to all supported operators and their
latest behavior. If you encounter issues with an older model, consider upgrading it using the ONNX
version converter.

### Upgrading ONNX Models

There are two simple ways to upgrade your ONNX models to the recommended opset version:

Option 1: Use the provided utility script:

```
uv run --script https://raw.githubusercontent.com/tracel-ai/burn/refs/heads/main/crates/burn-import/onnx_opset_upgrade.py
```

Option 2: Use a custom Python script:

```python
import onnx
from onnx import version_converter, shape_inference

# Load your ONNX model
model = onnx.load('path/to/your/model.onnx')

# Convert the model to opset version 16
upgraded_model = version_converter.convert_version(model, 16)

# Apply shape inference to the upgraded model
inferred_model = shape_inference.infer_shapes(upgraded_model)

# Save the converted model
onnx.save(inferred_model, 'upgraded_model.onnx')
```

## Step-by-Step Guide

Follow these steps to import an ONNX model into your Burn project:

### Step 1: Update `Cargo.toml`

First, add the required dependencies to your `Cargo.toml`:

```toml
[dependencies]
burn = { version = "~0.20", features = ["ndarray"] }
burn-store = { version = "~0.20", features = ["burnpack"] }

[build-dependencies]
burn-import = "~0.20"
```

The `burn-store` crate with the `burnpack` feature is required to load model weights at runtime.

### Step 2: Update `build.rs`

In your `build.rs` file:

```rust
use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

This generates Rust code and a `.burnpack` weights file from your ONNX model during the build process.

### Step 3: Modify `mod.rs`

In your `src/model/mod.rs` file, include the generated code:

```rust
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

### Step 4: Use the Imported Model

Now you can use the imported model in your code:

```rust
use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::my_model::Model;

fn main() {
    let device = NdArrayDevice::default();

    // Create model instance and load weights from target dir default device
    let model: Model<NdArray<f32>> = Model::default();

    // Create input tensor (replace with your actual input)
    let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 3, 224, 224], &device);

    // Perform inference
    let output = model.forward(input);

    println!("Model output: {:?}", output);
}
```

## Advanced Configuration

The `ModelGen` struct provides configuration options:

```rust
ModelGen::new()
    .input("path/to/model.onnx")
    .out_dir("model/")
    .development(true)   // Enable development mode for debugging
    .embed_states(true)  // Embed weights in the binary (for WASM)
    .run_from_script();
```

- `input`: Path to the ONNX model file
- `out_dir`: Output directory for generated code and weights
- `development`: When enabled, generates additional debug files (`.onnx.txt`, `.graph.txt`)
- `embed_states`: When enabled, embeds model weights in the binary using `include_bytes!`.
  Useful for WebAssembly or single-binary deployments. Not recommended for large models.

Model weights are stored in `.burnpack` format, which provides efficient serialization and loading.

## Loading and Using Models

You can load models in several ways:

```rust
// Load from the output directory with default device (recommended for most use cases)
// This automatically loads weights from the .burnpack file
let model = Model::<Backend>::default();

// Create a new model instance with a specific device
// (initializes weights randomly; load weights via `load_from` afterward)
let model = Model::<Backend>::new(&device);

// Load from a specific .burnpack file
let model = Model::<Backend>::from_file("path/to/weights.burnpack", &device);

// Load from embedded weights (if embed_states was true)
let model = Model::<Backend>::from_embedded(&device);
```

## Troubleshooting

Common issues and solutions:

1. **Unsupported ONNX operator**: Check the
   [list of supported ONNX operators](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).
   You may need to simplify your model or wait for support.

2. **Build errors**: Ensure your `burn-import` version matches your Burn version and verify the ONNX
   file path in `build.rs`.

3. **Runtime errors**: Confirm that your input tensors match the expected shape and data type of
   your model.

4. **Performance issues**: Consider using a more performant backend or optimizing your model
   architecture.

5. **Viewing generated files**: Find the generated Rust code and weights in the `OUT_DIR` directory
   (usually `target/debug/build/<project>/out`).

## Examples and Resources

For practical examples, check out:

1. [MNIST Inference Example](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference)
2. [SqueezeNet Image Classification](https://github.com/tracel-ai/models/tree/main/squeezenet-burn)

These demonstrate real-world usage of ONNX import in Burn projects.

## Conclusion

Importing ONNX models into Burn combines the vast ecosystem of pre-trained models with Burn's
performance and Rust's safety features. Following this guide, you can seamlessly integrate ONNX
models into your Burn projects for inference, fine-tuning, or further development.

The `burn-import` crate is actively developed, with ongoing work to support more ONNX operators and
improve performance. Stay tuned to the Burn repository for updates!

---

> 🚨**Note**: The `burn-import` crate is in active development. For the most up-to-date information
> on supported ONNX operators, please refer to the
> [official documentation](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).
