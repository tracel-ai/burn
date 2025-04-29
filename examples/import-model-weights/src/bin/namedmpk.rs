//! # Named MessagePack Model Weight Example
//!
//! This example demonstrates loading model weights from Burn's native Named MessagePack format
//! and using them for inference. It allows you to:
//!
//! 1. Load a model from a previously converted .mpk file
//! 2. Run inference on a specified MNIST test image
//! 3. Verify the prediction matches the expected label
//!
//! ## Usage
//!
//! ```shell
//! cargo run --bin namedmpk -- <image_index> <model_path>
//! ```
//!
//! Where:
//! - `<image_index>`: The index of the MNIST test image to classify (0-9999)
//! - `<model_path>`: Path to the model file (without .mpk extension)
//!
//! ## Example
//!
//! ```shell
//! cargo run --bin namedmpk -- 35 /tmp/burn-convert/mnist
//! ```
//!
//! This example is typically used after converting a model with the `convert` binary.

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
