//! # Model Weight Converter
//!
//! This tool converts model weights from PyTorch or Safetensors format into
//! Burn's native Named MessagePack format (.mpk).
//!
//! ## Usage
//!
//! ```shell
//! cargo run --bin convert -- <pytorch|safetensors> <output_directory>
//! ```
//!
//! Where:
//! - `<pytorch|safetensors>`: The source format to convert from
//! - `<output_directory>`: Directory where the converted .mpk file will be saved
//!
//! ## Examples
//!
//! ```shell
//! # Convert from PyTorch format
//! cargo run --bin convert -- pytorch /tmp/burn-convert
//!
//! # Convert from Safetensors format
//! cargo run --bin convert -- safetensors /tmp/burn-convert
//! ```
//!
//! ## Features
//!
//! - Supports PyTorch (.pt) and Safetensors (.safetensors) input formats
//! - Converts to Burn's native MPK format for efficient loading
//! - Preserves full precision of the original weights
//! - Creates a file named `mnist.mpk` in the specified output directory
//!
//! This tool is useful in the model deployment pipeline, allowing models trained
//! in popular frameworks to be used within Burn applications.
use std::{env, path::Path, process};

use burn::{
    backend::NdArray,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::PyTorchFileRecorder;
use burn_import::safetensors::SafetensorsFileRecorder;
use import_model_weights::ModelRecord;

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

    // Load the model record based on the specified format
    let model_record: ModelRecord<B> = match weight_format {
        "pytorch" => {
            println!("Loading PyTorch weights from '{}'...", PYTORCH_WEIGHTS_PATH);
            PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load(PYTORCH_WEIGHTS_PATH.into(), &device)
                .unwrap_or_else(|err| {
                    eprintln!(
                        "Error: Failed to load PyTorch model weights from '{}': {}",
                        PYTORCH_WEIGHTS_PATH, err
                    );
                    process::exit(1);
                })
        }
        "safetensors" => {
            println!(
                "Loading Safetensors weights from '{}'...",
                SAFETENSORS_WEIGHTS_PATH
            );
            SafetensorsFileRecorder::<FullPrecisionSettings>::default()
                .load(SAFETENSORS_WEIGHTS_PATH.into(), &device)
                .unwrap_or_else(|err| {
                    eprintln!(
                        "Error: Failed to load Safetensors model weights from '{}': {}",
                        SAFETENSORS_WEIGHTS_PATH, err
                    );
                    process::exit(1);
                })
        }
        _ => {
            eprintln!(
                "Error: Unsupported weight format '{}'. Please use 'pytorch' or 'safetensors'.",
                weight_format
            );
            process::exit(1);
        }
    };

    // Create a recorder for saving the model record in Burn's format
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    // Define the output path for the Burn model file
    let output_file_path = output_directory.join(MODEL_OUTPUT_NAME);

    println!(
        "Saving model record to '{}.mpk'...",
        output_file_path.display()
    );

    // Save the loaded record to the specified file path
    recorder
        .record(model_record, output_file_path.clone())
        .unwrap_or_else(|err| {
            eprintln!(
                "Error: Failed to save model record to '{}.mpk': {}",
                output_file_path.display(),
                err
            );
            process::exit(1);
        });

    println!(
        "Model record successfully saved to '{}.mpk'.",
        output_file_path.display()
    );
}
