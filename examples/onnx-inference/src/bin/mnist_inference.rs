use std::env::args;

use burn::{
    backend::ndarray::NdArray,
    data::dataset::{Dataset, vision::MnistDataset},
    tensor::Tensor,
};

use onnx_inference::mnist::Model;

const IMAGE_INX: usize = 42; // <- Change this to test a different image

fn main() {
    // Get image index argument (first) from command line

    let image_index = if let Some(image_index) = args().nth(1) {
        println!("Image index: {}", image_index);
        image_index
            .parse::<usize>()
            .expect("Failed to parse image index")
    } else {
        println!("No image index provided; Using default image index: {IMAGE_INX}");
        IMAGE_INX
    };

    assert!(image_index < 10000, "Image index must be less than 10000");

    type Backend = NdArray<f32>;

    // Get a default device for the backend
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    // Load the MNIST dataset and get an item
    let dataset = MnistDataset::test();
    let item = dataset.get(image_index).unwrap();

    // Create a tensor from the image data
    let image_data = item.image.iter().copied().flatten().collect::<Vec<f32>>();
    let mut input =
        Tensor::<Backend, 1>::from_floats(image_data.as_slice(), &device).reshape([1, 1, 28, 28]);

    // Normalize the input
    input = ((input / 255) - 0.1307) / 0.3081;

    // Run the model on the input
    let output = model.forward(input);

    // Get the index of the maximum value
    let arg_max = output.argmax(1).into_scalar() as u8;

    // Check if the index matches the label
    assert!(arg_max == item.label);

    println!("Success!");
    println!("Predicted: {}", arg_max);
    println!("Actual: {}", item.label);
    println!("See the image online, click the link below:");
    println!("https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test?row={image_index}");
}
