//! # PyTorch Model Weight Import Example
//!
//! This example demonstrates how to import PyTorch model weights directly into a Burn model.
//! It performs the following operations:
//! 1. Loads pre-trained PyTorch weights from a file (`weights/mnist.pt`)
//! 2. Converts them to a Burn model record
//! 3. Runs inference using the imported model on a specified MNIST test image
//!
//! ## Usage
//!
//! ```shell
//! cargo run --bin pytorch -- <image_index>
//! ```
//!
//! Where:
//! - `<image_index>`: The index of the MNIST test image to classify (0-9999)
//!   If omitted, a default index (42) will be used
//!
//! ## Example
//!
//! ```shell
//! cargo run --bin pytorch -- 15
//! ```
//!
//! This example uses the `PyTorchFileRecorder` from `burn_import` to seamlessly
//! load PyTorch weights into a Burn model without requiring an intermediate conversion step.

use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, Recorder};

use burn_import::pytorch::PyTorchFileRecorder;

use import_model_weights::{ModelRecord, infer};

type B = NdArray<f32>;

const WEIGHTS_FILE: &str = "weights/mnist.pt";

pub fn main() {
    println!("Loading PyTorch model weights from file: {}", WEIGHTS_FILE);

    // Load PyTorch weights into a model record.
    let record: ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(WEIGHTS_FILE.into(), &Default::default())
        .expect("Failed to load PyTorch model weights");

    // Infer using the loaded model record.
    infer(record);
}
