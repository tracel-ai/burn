use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, Recorder};

use burn_import::pytorch::PyTorchFileRecorder;

use import_model_weights::{ModelRecord, infer};

type B = NdArray<f32>;

const WEIGHTS_FILE: &str = "weights/mnist.pt";

pub fn main() {
    println!("Loading PyTorch model weights from file: {WEIGHTS_FILE}");

    // Load PyTorch weights into a model record.
    let record: ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(WEIGHTS_FILE.into(), &Default::default())
        .expect("Failed to load PyTorch model weights");

    // Infer using the loaded model record.
    infer(record);
}
