# Burn Import

`burn-import` is a crate designed to simplify the process of importing models trained in other
machine learning frameworks into the Burn framework. This tool generates a Rust source file that
aligns the imported model with Burn's model and converts tensor data into a format compatible with
Burn.

Currently, `burn-import` supports importing ONNX models with a limited set of operators, as it is
still under development.

## Supported ONNX Operators

- BatchNorm
- Conv2d
- Flatten
- Gemm (Linear layer)
- LogSoftmax

## Usage

### Importing ONNX models

To import ONNX models, follow these steps:

1. Add the following code to your `build.rs` file:

   ```rust
   use burn_import::onnx::ModelCodeGen;

   fn main() {
       ModelCodeGen::new()
           .input("src/model/mnist.onnx") // Path to the ONNX model
           .out_dir("model/")            // Directory for the generated Rust source file (under target/)
           .run_from_script();
   }
   ```

2. Add the following code to the `mod.rs` file under `src/model`:

   ```rust
   pub mod mnist {
       include!(concat!(env!("OUT_DIR"), "/model/mnist.rs"));
   }
   ```

3. Use the imported model in your code as shown below:

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

A working example can be found in the [`examples/onnx-inference`](https://github.com/burn-rs/burn/tree/main/examples/onnx-inference) directory.

### Adding new operators

To add support for new operators to `burn-import`, follow these steps:

1. Optimize the ONNX model using [onnxoptimizer](https://github.com/onnx/optimizer). This will
   remove unnecessary operators and constants, making the model easier to understand.
2. Use the [Netron](https://github.com/lutzroeder/netron) app to visualize the ONNX model.
3. Generate artifact files for the ONNX model (`my-model.onnx`) and its components:
   ```
   cargo r -- ./my-model.onnx ./
   ```
4. Implement the missing operators when you encounter an error stating that the operator is not
   supported. Ideally, the `my-model.graph.txt` file is generated before the error occurs, providing
   information about the ONNX model.
5. The newly generated `my-model.graph.txt` file contains IR information about the model, while the
   `my-model.rs` file contains an actual Burn model in Rust code. The `my-model.json` file contains
   the model data.
6. The `srs/onnx` directory contains the following ONNX modules (continued):

   - `coalesce.rs`: Coalesces multiple ONNX operators into a single Burn operator. This is useful
     for operators that are not supported by Burn but can be represented by a combination of
     supported operators.
   - `op_configuration.rs`: Contains helper functions for configuring Burn operators from operator
     nodes.
   - `shape_inference.rs`: Contains helper functions for inferring shapes of tensors for inputs and
     outputs of operators.

7. Add unit tests for the new operator in the `burn-import/tests/onnx_tests.rs` file. Add the ONNX
   file and expected output to the `tests/data` directory. Ensure the ONNX file is small, as large
   files can increase repository size and make it difficult to maintain and clone. Refer to existing
   unit tests for examples.

## Resources

1. [PyTorch to ONNX](https://pytorch.org/docs/stable/onnx.html)
2. [ONNX to Pytorch](https://github.com/ENOT-AutoDL/onnx2torch)
3. [ONNX Intro](https://onnx.ai/onnx/intro/)
4. [ONNX Operators](https://onnx.ai/onnx/operators/index.html)
5. [ONNX Protos](https://onnx.ai/onnx/api/classes.html)
6. [ONNX Optimizer](https://github.com/onnx/optimizer)
7. [Netron](https://github.com/lutzroeder/netron)
