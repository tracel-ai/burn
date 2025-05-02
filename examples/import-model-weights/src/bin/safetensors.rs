//! # Safetensors Model Weight Import Example
//!
//! This example demonstrates importing pre-trained model weights from Safetensors format
//! directly into a Burn ML model. It performs the following operations:
//!
//! 1. Loads model weights from a Safetensors file (`weights/mnist.safetensors`) that was previously exported from PyTorch
//! 2. Constructs a ModelRecord with those weights
//! 3. Uses the loaded model to perform inference on a specified MNIST test image
//!
//! ## Usage
//!
//! ```shell
//! cargo run --bin safetensors -- <image_index>
//! ```
//!
//! Where:
//! - `<image_index>`: The index of the MNIST test image to classify (0-9999)
//!   If omitted, a default index (42) will be used
//!
//! ## Example
//!
//! ```shell
//! cargo run --bin safetensors -- 42
//! ```
//!
//! This example uses the `SafetensorsFileRecorder` from `burn_import` to seamlessly
//! load Safetensors weights into a Burn model without requiring an intermediate conversion step.
//! Safetensors is a safer and more efficient alternative to PyTorch's native format.

use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, Recorder};

use burn_import::safetensors::SafetensorsFileRecorder;

use import_model_weights::{ModelRecord, infer};

type B = NdArray<f32>;

const WEIGHTS_FILE: &str = "weights/mnist.safetensors";

pub fn main() {
    println!(
        "Loading Safetensors model weights from file: {}",
        WEIGHTS_FILE
    );
    // Load Safetensors weights exported from PyTorch into a model record.
    let record: ModelRecord<B> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(WEIGHTS_FILE.into(), &Default::default())
        .expect("Failed to load Safetensors model weights");

    // Infer using the loaded model record.
    infer(record);
}
