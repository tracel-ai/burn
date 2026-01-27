# ONNX Import

Burn supports importing [ONNX (Open Neural Network Exchange)](https://onnx.ai/onnx/intro/index.html)
models through the [`burn-onnx`](https://github.com/tracel-ai/burn-onnx) crate, enabling you to use
pre-trained models from PyTorch, TensorFlow, and other frameworks in your Rust projects.

## Why Import ONNX Models?

- **Time-saving**: Skip the resource-intensive process of training models from scratch
- **Access to state-of-the-art architectures**: Utilize cutting-edge models from researchers and
  industry
- **Transfer learning**: Fine-tune imported models for your specific tasks
- **Native Rust code generation**: Models are converted to Rust source code, not interpreted at
  runtime
- **No runtime dependency**: No ONNX runtime needed
- **Trainability**: Imported models can be further trained using Burn

## Quick Start

### Step 1: Add Dependencies

```toml
[dependencies]
burn = { version = "~0.21", features = ["ndarray"] }

[build-dependencies]
burn-onnx = "~0.21"
```

### Step 2: Create `build.rs`

```rust
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

### Step 3: Include Generated Code

In `src/model/mod.rs`:

```rust
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

### Step 4: Use the Model

```rust
use burn_ndarray::{NdArray, NdArrayDevice};
use model::my_model::Model;

fn main() {
    let device = NdArrayDevice::default();

    // Load model with weights
    let model: Model<NdArray<f32>> = Model::default();

    // Create input tensor
    let input = burn::tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 3, 224, 224], &device);

    // Run inference
    let output = model.forward(input);
    println!("Output: {:?}", output);
}
```

## Configuration Options

```rust
ModelGen::new()
    .input("path/to/model.onnx")
    .out_dir("model/")
    .development(true)   // Generate debug files (.onnx.txt, .graph.txt)
    .embed_states(true)  // Embed weights in binary (useful for WASM)
    .run_from_script();
```

## Loading Models

```rust
// Default: load from output directory
let model = Model::<Backend>::default();

// From specific file
let model = Model::<Backend>::from_file("path/to/weights.burnpack", &device);

// From embedded weights (if embed_states was true)
let model = Model::<Backend>::from_embedded(&device);
```

## ONNX Compatibility

Burn recommends **opset version 16 or higher**. To upgrade older models:

```bash
uv run --script https://raw.githubusercontent.com/tracel-ai/burn-onnx/refs/heads/main/onnx_opset_upgrade.py
```

## Resources

- [burn-onnx Repository](https://github.com/tracel-ai/burn-onnx) - Source code and issues
- [Supported ONNX Operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md) -
  Complete list of supported operators
- [Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md) -
  Contributing new operators
- [MNIST Inference Example](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference) -
  Working example

## Troubleshooting

1. **Unsupported operator**: Check the
   [supported operators list](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md)

2. **Build errors**: Ensure `burn-onnx` version matches your Burn version

3. **Runtime errors**: Verify input tensor shapes match model expectations

4. **Finding generated files**: Look in `target/debug/build/<project>/out/model/`
