# ONNX Import

## Introduction

As deep learning evolves, interoperability between frameworks becomes crucial. Burn provides robust
support for importing [ONNX (Open Neural Network Exchange)](https://onnx.ai/onnx/intro/index.html)
models through the [`burn-onnx`](https://github.com/tracel-ai/burn-onnx) crate, enabling you to
leverage pre-trained models in your Rust-based deep learning projects.

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
uv run --script https://raw.githubusercontent.com/tracel-ai/burn-onnx/refs/heads/main/onnx_opset_upgrade.py
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
burn = { version = "~0.21", features = ["ndarray"] }

[build-dependencies]
burn-onnx = "~0.21"
```

### Step 2: Update `build.rs`

In your `build.rs` file:

```rust, ignore
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

This generates Rust code and a `.bpk` weights file from your ONNX model during the build process.

### Step 3: Modify `mod.rs`

In your `src/model/mod.rs` file, include the generated code:

```rust, ignore
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

### Step 4: Use the Imported Model

Now you can use the imported model in your code:

```rust, ignore
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

```rust, ignore
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
- `embed_states`: When enabled, embeds model weights in the binary using `include_bytes!`. Useful
  for WebAssembly or single-binary deployments. Not recommended for large models.

Model weights are stored in Burnpack format (`.bpk`), which provides efficient serialization and
loading.

## Loading and Using Models

You can load models in several ways:

```rust, ignore
// Load from the output directory with default device (recommended for most use cases)
// This automatically loads weights from the .bpk file
let model = Model::<Backend>::default();

// Create a new model instance with a specific device
// (initializes weights randomly; load weights via `load_from` afterward)
let model = Model::<Backend>::new(&device);

// Load from a specific .bpk file
let model = Model::<Backend>::from_file("path/to/weights.bpk", &device);

// Load from embedded weights (if embed_states was true)
let model = Model::<Backend>::from_embedded(&device);
```

## Troubleshooting

Common issues and solutions:

1. **Unsupported ONNX operator**: Check the
   [list of supported ONNX operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md).
   You may need to simplify your model or wait for support.

2. **Build errors**: Ensure your `burn-onnx` version matches your Burn version and verify the ONNX
   file path in `build.rs`.

3. **Runtime errors**: Confirm that your input tensors match the expected shape and data type of
   your model.

4. **Performance issues**: Consider using a more performant backend or optimizing your model
   architecture.

5. **Viewing generated files**: Find the generated Rust code and weights in the `OUT_DIR` directory
   (usually `target/debug/build/<project>/out`).

## Examples and Resources

For practical examples, check out the
[burn-onnx examples](https://github.com/tracel-ai/burn-onnx/tree/main/examples):

1. [ONNX Inference](https://github.com/tracel-ai/burn-onnx/tree/main/examples/onnx-inference) -
   MNIST inference example
2. [Image Classification Web](https://github.com/tracel-ai/burn-onnx/tree/main/examples/image-classification-web) -
   SqueezeNet running in the browser via WebAssembly
3. [Raspberry Pi Pico](https://github.com/tracel-ai/burn-onnx/tree/main/examples/raspberry-pi-pico) -
   Embedded deployment example

These demonstrate real-world usage of ONNX import in Burn projects.

For contributors looking to add support for new ONNX operators:

- [Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md) -
  Step-by-step guide for implementing new operators

## Conclusion

Importing ONNX models into Burn combines the vast ecosystem of pre-trained models with Burn's
performance and Rust's safety features. Following this guide, you can seamlessly integrate ONNX
models into your Burn projects for inference, fine-tuning, or further development.

The `burn-onnx` crate is actively developed, with ongoing work to support more ONNX operators and
improve performance. Visit the [burn-onnx repository](https://github.com/tracel-ai/burn-onnx) for
updates and to contribute!
