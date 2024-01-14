/// This build script does the following:
/// 1. Loads PyTorch weights into a model record.
/// 2. Saves the model record to a file using the `NamedMpkFileRecorder`.
///
/// The model source code is included directly in this build script because
/// it cannot be imported from the crate directly.
use std::env;
use std::path::Path;

use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;

// Include the source code of the model since we cannot import from the crate directly.
mod model {
    #![allow(dead_code)]
    include!("src/model/mnist.rs");
}

// Basic backend type (not used directly here).
type B = NdArray<f32>;

fn main() {
    // Load PyTorch weights into a model record.
    let record: model::ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load("pytorch/mnist.pt".into())
        .expect("Failed to decode state");

    // Save the model record to a file.
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    // Save into the OUT_DIR directory so that the model can be loaded by the
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/mnist");

    recorder
        .record(record, file_path)
        .expect("Failed to save model record");
}
