# Burn Import

`burn-import` is a crate designed to facilitate importing models trained in other machine learning
frameworks into the Burn framework. This tool generates a Rust source file that aligns the source
model with Burn's model and converts tensor data into a format compatible with Burn.

Currently under development, `burn-import` supports importing ONNX models with a limited set of
operators.

## Supported ONNX Operators

- BatchNorm
- Conv2d
- Flatten
- Gemm (Linear layer)
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

### Adding new operators

This section explains how to add support for new operators to `burn-import`.

1. Optimize the ONNX model using [onnxoptimizer](https://github.com/onnx/optimizer). It will remove
   uncessary operator/constants and make the model easier to understand.
2. Use [Netron](https://github.com/lutzroeder/netron) app to visualize the ONNX model.
3. Generate artifact files to help to see what the ONNX model (`my-model.onnx) looks like and its
   components.
   ```bash
   cargo r -- ./my-model.onnx ./
   ```
4. You will run into an error saying that the operator is not supported. Implement missing
   operators. Hopefully, at least `my-model.graph.txt` is generated before the error occurs. This
   file contains information about the ONNX model.
5. The newly generated `my-model.graph.txt` file will contain IR information about the model. This
   file is useful for understanding the structure of the model and the operators it uses. The
   `my-model.rs` file will contain an actual Burn model in rust code. `my-model.json` will contain
   the data of the model.
6. The following is the explaination of onnx modules (under `srs/onnx`):
   - `from_onnx.rs`: This module contains logic for converting ONNX data objects into IR
     (Intermediate Representation) objects. This module must contain anything that deals with ONNX
     directly.
   - `ir.rs`: This module contains the IR objects that are used to represent the ONNX model. These
     objects are used to generate the Burn model.
   - `to_burn.rs` - This module contains logic for converting IR objects into Burn model source code
     and data. Nothing in this module should deal with ONNX directly.
   - `coalesce.rs`: This module contains the logic to coalesce multiple ONNX operators into a single
     Burn operator. This is useful for operators that are not supported by Burn, but can be
     represented by a combination of supported operators.
   - `op_configuration.rs` - This module contains helper functions for configuring burn operators
     from operator nodes.
   - `shape_inference.rs` - This module contains helper functions for inferring shapes of tensors
     for inputs and outputs of operators.
7. Add unit tests for the new operator in `burn-import/tests/onnx_tests.rs` file. Add the ONNX file
   and expected output to `tests/data` directory. Please be sure the ONNX file is small. If the ONNX
   file is too large, the repository size will grow too large and will be difficult to maintain and
   clone. See the existing unit tests for examples.

## Resources

1. [PyTorch ONNX](https://pytorch.org/docs/stable/onnx.html)
2. [ONNX Intro](https://onnx.ai/onnx/intro/)
