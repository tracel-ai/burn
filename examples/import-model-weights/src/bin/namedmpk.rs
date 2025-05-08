use std::env;
use std::path::Path;

use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

use import_model_weights::{ModelRecord, infer};

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
        "Loading model weights from file: {}.mpk",
        model_path.display()
    );

    // Load the model record from the specified path
    let record: ModelRecord<B> = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(model_path.into(), &Default::default())
        .expect("Failed to decode state from specified path");

    // Infer using the loaded model record.
    infer(record);
}
