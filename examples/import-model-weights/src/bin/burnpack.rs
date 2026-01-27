use std::env;
use std::path::Path;

use burn::backend::NdArray;

use burn_store::{BurnpackStore, ModuleSnapshot};
use import_model_weights::{Model, infer};

type B = NdArray<f32>;

pub fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <image_index> <model_path>", args[0]);
        std::process::exit(1);
    }

    let model_path_str = &args[2];
    let model_path = Path::new(model_path_str);
    println!(
        "Loading model weights from file: {}.bpk",
        model_path.display()
    );

    // Initialize a model with default weights
    let device = Default::default();
    let mut model: Model<B> = Model::init(&device);

    // Load the model from the Burnpack file
    let mut store = BurnpackStore::from_file(model_path);
    model
        .load_from(&mut store)
        .expect("Failed to load model from Burnpack file");

    // Infer using the loaded model
    infer(model);
}
