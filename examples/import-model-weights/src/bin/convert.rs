use std::{env, path::Path, process};

use burn::backend::NdArray;
use burn_store::{
    BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use import_model_weights::Model;

// Path constants
const PYTORCH_WEIGHTS_PATH: &str = "weights/mnist.pt";
const SAFETENSORS_WEIGHTS_PATH: &str = "weights/mnist.safetensors";
const MODEL_OUTPUT_NAME: &str = "mnist";

// Basic backend type (not used for computation).
type B = NdArray<f32>;

pub fn main() {
    let args: Vec<String> = env::args().collect();

    // Check argument count
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <pytorch|safetensors> <output_directory>",
            args[0]
        );
        process::exit(1);
    }

    // Get weight format and output directory from arguments
    let weight_format = args[1].as_str();
    let output_directory = Path::new(&args[2]);

    // Use the default device (CPU)
    let device = Default::default();

    // Initialize a model with default weights
    let mut model: Model<B> = Model::init(&device);

    // Load the model weights based on the specified format
    match weight_format {
        "pytorch" => {
            println!("Loading PyTorch weights from '{PYTORCH_WEIGHTS_PATH}'...");
            let mut store = PytorchStore::from_file(PYTORCH_WEIGHTS_PATH);
            model.load_from(&mut store).unwrap_or_else(|e| {
                panic!("Failed to load PyTorch model weights from '{PYTORCH_WEIGHTS_PATH}': {e}")
            });
        }
        "safetensors" => {
            println!("Loading Safetensors weights from '{SAFETENSORS_WEIGHTS_PATH}'...");
            let mut store = SafetensorsStore::from_file(SAFETENSORS_WEIGHTS_PATH)
                .with_from_adapter(PyTorchToBurnAdapter);
            model.load_from(&mut store).unwrap_or_else(|e| {
                panic!(
                    "Failed to load Safetensors model weights from '{SAFETENSORS_WEIGHTS_PATH}': {e}"
                )
            });
        }
        _ => {
            eprintln!(
                "Error: Unsupported weight format '{weight_format}'. Please use 'pytorch' or 'safetensors'."
            );
            process::exit(1);
        }
    };

    // Define the output path for the Burn model file
    let output_file_path = output_directory.join(MODEL_OUTPUT_NAME);

    println!("Saving model to '{}.bpk'...", output_file_path.display());

    // Save the model using BurnpackStore
    let mut store = BurnpackStore::from_file(&output_file_path).overwrite(true);
    model.save_into(&mut store).unwrap_or_else(|e| {
        panic!(
            "Failed to save model to '{}.bpk': {e}",
            output_file_path.display()
        )
    });

    println!(
        "Model successfully saved to '{}.bpk'.",
        output_file_path.display()
    );
}
