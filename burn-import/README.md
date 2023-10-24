# Burn Import: A Crate for ONNX Model Import into the Burn Framework

`burn-import` facilitates the seamless import of machine learning models, particularly those in the
ONNX format, into the Burn deep learning framework. It automatically generates Rust source code,
aligns the model structure with Burn's native format, and converts tensor data for Burn
compatibility.

> **Note**: This crate is in active development and currently supports a
> [limited set of ONNX operators](SUPPORTED-ONNX-OPS.md).

## Working Examples

For practical examples, please refer to:

1. [ONNX Inference Example](https://github.com/burn-rs/burn/tree/main/examples/onnx-inference)
2. [SqueezeNet Image Classification](https://github.com/burn-rs/models/tree/main/squeezenet-burn)

## Usage

### Importing ONNX Models

Follow these steps to import an ONNX model into your Burn project:

1. **Update `build.rs`**: Include the following Rust code in your `build.rs` file:

   ```rust
   use burn_import::onnx::ModelGen;

   fn main() {
       // Generate Rust code from the ONNX model file
       ModelGen::new()
           .input("src/model/model_name.onnx")
           .out_dir("model/")
           .run_from_script();
   }
   ```

2. **Modify `mod.rs`**: Add this code to the `mod.rs` file located in `src/model`:

   ```rust
   pub mod model_name {
       include!(concat!(env!("OUT_DIR"), "/model/model_name.rs"));
   }
   ```

3. **Utilize Imported Model**: Use the following sample code to incorporate the imported model into
   your application:

   ```rust
   mod model;

   use burn::tensor;
   use burn_ndarray::NdArrayBackend;
   use model::model_name::Model;

   fn main() {
       // Initialize a new model instance
       let model: Model<NdArrayBackend<f32>> = Model::new();

       // Create a sample input tensor (zeros for demonstration)
       let input = tensor::Tensor::<NdArrayBackend<f32>, 4>::zeros([1, 1, 28, 28]);

       // Execute the model
       let output = model.forward(input);

       // Display the output
       println!("{:?}", output);
   }
   ```

## Contribution

Interested in contributing to `burn-import`? Check out our [development guide](DEVELOPMENT.md) for
more information.
