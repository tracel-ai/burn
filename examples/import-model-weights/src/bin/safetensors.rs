use burn::backend::NdArray;

use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use import_model_weights::{Model, infer};

type B = NdArray<f32>;

const WEIGHTS_FILE: &str = "weights/mnist.safetensors";

pub fn main() {
    println!("Loading Safetensors model weights from file: {WEIGHTS_FILE}");

    // Initialize a model with default weights
    let device = Default::default();
    let mut model: Model<B> = Model::init(&device);

    // Load Safetensors weights into the model (using PyTorch adapter since weights were exported from PyTorch)
    let mut store =
        SafetensorsStore::from_file(WEIGHTS_FILE).with_from_adapter(PyTorchToBurnAdapter);
    model
        .load_from(&mut store)
        .expect("Failed to load Safetensors model weights");

    // Infer using the loaded model
    infer(model);
}
