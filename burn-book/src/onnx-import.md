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
   integration with Burn's ecosystem. The generated code is readable and can be edited by hand.
2. **Compile-time optimization**: Leverages the Rust compiler to optimize the generated code, and
   simplifies the graph (constant folding, shape propagation, dead code elimination) before
   generating it.
3. **No runtime dependency**: Eliminates the need for an ONNX runtime, unlike many other solutions.
4. **Trainability**: Allows imported models to be further trained or fine-tuned using Burn.
5. **Portability**: Enables compilation for various targets, including WebAssembly and `no_std`
   embedded devices.
6. **Backend flexibility**: Works with any of Burn's supported backends.

## ONNX Compatibility

`burn-onnx` supports ONNX opset versions 1 through 24: every supported operator handles each opset it
exists in, including attributes that later became inputs and defaults that changed between
versions. Models can be imported as they are, without upgrading them first. The list of
[supported ONNX operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md)
covers the standard operator set; operators outside it can be supplied as
[custom operators](#custom-operators).

## Step-by-Step Guide

Follow these steps to import an ONNX model into your Burn project:

### Step 1: Update `Cargo.toml`

First, add the required dependencies to your `Cargo.toml`:

```toml
[dependencies]
burn = { version = "~0.22", features = ["flex"] }
burn-store = "~0.22"

[build-dependencies]
burn-onnx = "~0.22"
```

The generated code loads its weights through `burn-store`, so it must be a regular dependency of
your crate. `burn` needs at least one backend feature; `flex` is the portable CPU backend.

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
Call `.input()` once per model to convert several models at once.

### Step 3: Modify `mod.rs`

In your `src/model/mod.rs` file, include the generated code. The generated file is named after the
ONNX file:

```rust, ignore
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

### Step 4: Use the Imported Model

Now you can use the imported model in your code:

```rust, ignore
use burn::tensor::{Device, Tensor};
use model::my_model::Model;

fn main() {
    let device = Device::flex();

    // Create the model and load the weights written by the build script
    let model = Model::from_file(concat!(env!("OUT_DIR"), "/model/my_model.bpk"), &device);

    // Create input tensor (replace with your actual input)
    let input = Tensor::<4>::zeros([1, 3, 224, 224], &device);

    // Perform inference
    let output = model.forward(input);

    println!("Model output: {output}");
}
```

The generated `Model` is an ordinary Burn `Module` with a typed `forward` method: inputs and outputs
are `Tensor<D>` values (or plain scalars) in the order the ONNX graph declares them.

## Inspecting the Generated Code

The generated `.rs` file is regular Burn code, and reading it is the quickest way to understand or
debug an import. In a build script setup it lives in Cargo's `OUT_DIR`, typically
`target/debug/build/<your-crate>-<hash>/out/model/`.

To generate it somewhere easier to browse, use the `onnx2burn` command line tool:

```sh
cargo install burn-onnx
onnx2burn path/to/my_model.onnx ./generated
```

This writes `my_model.rs`, `my_model.bpk`, and `my_model.onnx.txt` (a dump of the parsed graph) to
`./generated`. You can also check the generated code into your project this way and edit it, instead
of regenerating it on every build.

## Advanced Configuration

The `ModelGen` struct provides configuration options:

```rust, ignore
use burn_onnx::{ModelGen, LoadStrategy};

ModelGen::new()
    .input("path/to/model.onnx")
    .out_dir("model/")
    .development(true)                       // Also write a debug dump of the parsed graph
    .load_strategy(LoadStrategy::Embedded)   // Embed weights in the binary
    .run_from_script();
```

- `input`: Path to the ONNX model file. Call it again to convert several models.
- `out_dir`: Output directory for generated code and weights, relative to `OUT_DIR`
- `development`: When enabled, also writes `<model>.onnx.txt`, a dump of the parsed ONNX graph with
  inferred types
- `load_strategy`: Controls which weight-loading constructors are generated on the `Model` struct
  (see below)
- `simplify`: Graph simplification before code generation (default: `true`)
- `partition`: Splits graphs with more than 200 nodes into submodules so the generated code stays
  quick to compile (default: `true`)
- `register_custom_op` / `register_op_override`: Supply code for operators `burn-onnx` does not
  support, or replace the code generated for one it does (see
  [Custom Operators](#custom-operators))

Use `run_from_script()` in a `build.rs` and `run_from_cli()` from a regular program, where
`out_dir` is used as a plain path.

Model weights are stored in Burnpack format (`.bpk`), which provides efficient serialization and
loading.

### Load Strategy

The `LoadStrategy` enum controls how the generated model loads its weights:

| Strategy   | Generated constructors              | `Default` impl | Use case                          |
| ---------- | ----------------------------------- | -------------- | --------------------------------- |
| `File`     | `from_file()`, `from_bytes()`       | Yes            | Standard desktop/server (default) |
| `Embedded` | `from_embedded()`, `from_bytes()`   | Yes            | Single binary, small models       |
| `Bytes`    | `from_bytes()`                      | No             | WASM, embedded, custom loaders    |
| `None`     | (none)                              | No             | Manual weight management          |

The default strategy is `File`, which keeps weights in a separate `.bpk` file and generates a
`from_file()` constructor.

For WebAssembly or environments without filesystem access, use `LoadStrategy::Bytes`:

```rust, ignore
ModelGen::new()
    .input("model.onnx")
    .out_dir("model/")
    .load_strategy(LoadStrategy::Bytes)
    .run_from_script();
```

Then load weights at runtime from any byte source (e.g., a network fetch):

```rust, ignore
let model = Model::from_bytes(weight_bytes, &device);
```

## Loading and Using Models

You can load models in several ways, depending on the `LoadStrategy` used during code generation:

```rust, ignore
// Load from a specific .bpk file (LoadStrategy::File)
let model = Model::from_file("path/to/weights.bpk", &device);

// Load from in-memory bytes (LoadStrategy::File, Embedded, or Bytes)
let model = Model::from_bytes(weight_bytes, &device);

// Load from embedded weights (LoadStrategy::Embedded)
let model = Model::from_embedded(&device);

// Load with the default device (LoadStrategy::File or Embedded). With File, this reads the
// .bpk from the absolute OUT_DIR path captured at build time, which suits development but not
// a binary you distribute.
let model = Model::default();
```

`Model::new(&device)` also exists, but it only builds the module structure: layers get freshly
initialized parameters and ONNX constants are zero. Use it only when you load the weights yourself
afterward, for example with `load_from` and a `BurnpackStore`.

## Custom Operators

An ONNX model can contain operators `burn-onnx` does not support: operators from vendor domains
such as `com.microsoft`, custom operators emitted by a framework's exporter, or standard operators
not implemented yet. Instead of failing, the import lets you supply the code for them with hooks
registered on `ModelGen`:

```rust, ignore
ModelGen::new()
    .input("src/model/my_model.onnx")
    .out_dir("model/")
    .register_custom_op(FftReal)      // handles my_domain::FftReal
    .register_op_override(MyMatMul)   // replaces the generated code for every MatMul
    .run_from_script();
```

- A **`CustomOp`** provides type inference and code generation for one ONNX `(op_type, domain)`
  pair. It can read the node's attributes and constant inputs, and typically emits a call into an
  ordinary Rust function in your crate.
- An **`OpOverride`** replaces the code generated for a built-in operator, for example to route it
  to a fused, quantized, or hardware-specific kernel. Type inference still comes from the built-in
  operator.

Everything a hook needs is re-exported from `burn_onnx::ext`. To find out which operators a model is
missing, build it with no hooks registered: the error lists every unsupported operator, its domain,
and how many nodes use it.

The [custom-op-hooks](https://github.com/tracel-ai/burn-onnx/tree/main/examples/custom-op-hooks)
example shows both kinds of hook end to end.

## Exporting Burn Models to ONNX

`burn-onnx` can also go the other way. With the `export` feature enabled, `OnnxExporter` runs a
module's forward pass once, records the tensor operations, and writes them out as an ONNX model with
the weights embedded:

```toml
[dependencies]
burn-onnx = { version = "~0.22", features = ["export"] }
```

```rust, ignore
use burn_onnx::export::OnnxExporter;

let sample = Tensor::<4>::zeros([1, 3, 224, 224], &device);
OnnxExporter::new()
    .export(&model, sample, MyModel::forward)?
    .save("my_model.onnx")?;
```

`export` fixes every dimension to the sample input's shape. To keep an axis such as the batch size
dynamic, use `export_dynamic` with a second sample input and an `InputSpec` per input marking the
dynamic axes.

Export is experimental. It targets opset 18 and covers the operations common in convolutional and
fully connected networks; an operation it cannot lower yet is reported as
`ExportError::UnsupportedOperation`. See the
[`export` module documentation](https://docs.rs/burn-onnx/latest/burn_onnx/export/index.html) for
details.

## Troubleshooting

Common issues and solutions:

1. **Unsupported ONNX operator**: The build error lists every operator the model uses that has no
   implementation. Check the
   [list of supported ONNX operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md),
   then implement the missing ones as [custom operators](#custom-operators) or open an issue.

2. **Build errors**: Make sure `burn`, `burn-store`, and `burn-onnx` share the same version, that
   `burn-store` is listed under `[dependencies]`, and that the ONNX file path in `build.rs` is
   correct. If the generated code itself fails to compile, please report it with the model.

3. **Wrong outputs**: Make sure the model was created with `from_file`, `from_bytes`,
   `from_embedded`, or `default`, not `new`. Then compare against ONNX Runtime with the same input.

4. **Runtime errors**: Confirm that your input tensors match the expected shape and data type of
   your model.

5. **Performance issues**: Use a GPU backend for large models, and build in release mode.

6. **Viewing generated files**: Find the generated Rust code and weights in the `OUT_DIR` directory
   (usually `target/debug/build/<project>/out`), or generate them with `onnx2burn` as described in
   [Inspecting the Generated Code](#inspecting-the-generated-code).

## Examples and Resources

For practical examples, check out the
[burn-onnx examples](https://github.com/tracel-ai/burn-onnx/tree/main/examples):

1. [ONNX Inference](https://github.com/tracel-ai/burn-onnx/tree/main/examples/onnx-inference) -
   MNIST inference example
2. [Image Classification Web](https://github.com/tracel-ai/burn-onnx/tree/main/examples/image-classification-web) -
   SqueezeNet running in the browser via WebAssembly
3. [Raspberry Pi Pico](https://github.com/tracel-ai/burn-onnx/tree/main/examples/raspberry-pi-pico) -
   `no_std` inference on a microcontroller with embedded weights
4. [Custom Op Hooks](https://github.com/tracel-ai/burn-onnx/tree/main/examples/custom-op-hooks) -
   Importing a model with custom operators and overriding a built-in one

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
