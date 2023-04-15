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
