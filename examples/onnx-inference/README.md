# ONNX Inference

This crate provides a simple example for importing MNIST ONNX model to Burn. The onnx file is
converted into a Rust source file using `burn-import` and the weights are stored in and loaded from
a binary file.

## Usage

```bash
cargo run -- 15
```

Output:

```bash
Finished dev [unoptimized + debuginfo] target(s) in 0.13s
    Running `burn/target/debug/onnx-inference 15`

Image index: 15
Success!
Predicted: 5
Actual: 5
See the image online, click the link below:
https://datasets-server.huggingface.co/assets/mnist/--/mnist/test/15/image/image.jpg
```

## How to import

1. Create `model` directory under `src`
2. Copy the ONNX model to `src/model/mnist.onnx`
3. Add the following to `mod.rs`:
   ```rust
   pub mod mnist {
       include!(concat!(env!("OUT_DIR"), "/model/mnist.rs"));
   }
   ```
4. Add the module to `lib.rs`:

   ```rust
   pub mod model;

   pub use model::mnist::*;
   ```

5. Add the following to `build.rs`:

   ```rust
   use burn_import::onnx::ModelGen;

   fn main() {
       // Generate the model code from the ONNX file.
       ModelGen::new()
           .input("src/model/mnist.onnx")
           .out_dir("model/")
           .run_from_script();
   }

   ```

6. Add your model to `src/bin` as a new file, in this specific case we have
called it `mnist.rs`:

   ```rust
   use burn::tensor;
   use burn_ndarray::NdArrayBackend;

   use onnx_inference::mnist::Model;

   fn main() {
       // Create a new model and load the state
       let model: Model<Backend> = Model::new().load_state();

       // Create a new input tensor (all zeros for demonstration purposes)
       let input = tensor::Tensor::<NdArrayBackend<f32>, 4>::zeros([1, 1, 28, 28]);

       // Run the model
       let output = model.forward(input);

       // Print the output
       println!("{:?}", output);
   }
   ```

7. Run `cargo build` to generate the model code, weights, and `mnist` binary.

## How to export PyTorch model to ONNX

The following steps show how to export a PyTorch model to ONNX from checked in PyTorch code (see
`pytorch/mnist.py`).

1. Install dependencies:

   ```bash
   pip install torch torchvision onnx
   ```

2. Run the following script to run the MNIST training and export the model to ONNX:

   ```bash
   python3 pytorch/mnist.py
   ```

This will generate `pytorch/mnist.onnx`.

## Resources

1. [PyTorch ONNX](https://pytorch.org/docs/stable/onnx.html)
2. [ONNX Intro](https://onnx.ai/onnx/intro/)
