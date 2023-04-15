# Burn Import

`burn-import` is a crate designed to facilitate importing models trained in other machine learning
frameworks into the Burn framework. This tool generates a Rust source file that aligns the source
model with Burn's model and converts tensor data into a format compatible with Burn.

Currently under development, `burn-import` supports importing ONNX models with a limited set of
operators.

## Supported ONNX Operators

- Conv2d
- Gemm (Linear layer)
- Flatten
- LogSoftmax

## Usage

### Importing ONNX models

In `build.rs`, add the following:

```rust
use burn_import::onnx::ModelCodeGen;

fn main() {
    ModelCodeGen::new()
        .input("src/model/mnist.onnx") // Path to the ONNX model
        .out_dir("model/")            // Directory to output the generated Rust source file (under target/)
        .run_from_script();
}
```

Then, add the following to mod.rs under `src/model`:

```rust
pub mod mnist {
    include!(concat!(env!("OUT_DIR"), "/model/mnist.rs"));
}
```

Finally, in your code, you can use the imported model as follows:

```rust
use burn::tensor;
use burn_ndarray::NdArrayBackend;
use onnx_inference::model::mnist::{Model, INPUT1_SHAPE};

fn main() {

    // Create a new model
    let model: Model<NdArrayBackend<f32>> = Model::new();

    // Create a new input tensor (all zeros for demonstration purposes)
    let input = tensor::Tensor::<NdArrayBackend<f32>, 4>::zeros(INPUT1_SHAPE);

    // Run the model
    let output = model.forward(input);

    // Print the output
    println!("{:?}", output);
}
```

You can view the working example in the `examples/onnx-inference` directory.
