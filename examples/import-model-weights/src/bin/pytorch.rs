use burn::backend::NdArray;

use burn_store::{ModuleSnapshot, PytorchStore};

use import_model_weights::{Model, infer};

type B = NdArray<f32>;

const WEIGHTS_FILE: &str = "weights/mnist.pt";

pub fn main() {
    println!("Loading PyTorch model weights from file: {WEIGHTS_FILE}");

    // Initialize a model with default weights
    let device = Default::default();
    let mut model: Model<B> = Model::init(&device);

    // Load PyTorch weights into the model
    let mut store = PytorchStore::from_file(WEIGHTS_FILE);
    model
        .load_from(&mut store)
        .expect("Failed to load PyTorch model weights");

    // Infer using the loaded model
    infer(model);
}
