/// This build script does the following:
/// 1. Loads PyTorch weights into a model record.
/// 2. Saves the model record to a file using the `NamedMpkFileRecorder`.
use std::path::Path;

use burn::{
    backend::NdArray,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::PyTorchFileRecorder;

// Basic backend type (not used directly here).
type B = NdArray<f32>;

fn main() {
    let device = Default::default();

    // Load PyTorch weights into a model record.
    let record: model::ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load("pytorch/mnist.pt".into(), &device)
        .expect("Failed to decode state");

    // Save the model record to a file.
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    // Save into the OUT_DIR directory so that the model can be loaded by the
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/mnist");

    recorder
        .record(record, file_path)
        .expect("Failed to save model record");
}
