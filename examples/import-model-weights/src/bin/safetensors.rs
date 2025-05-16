use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, Recorder};

use burn_import::safetensors::SafetensorsFileRecorder;

use import_model_weights::{ModelRecord, infer};

type B = NdArray<f32>;

const WEIGHTS_FILE: &str = "weights/mnist.safetensors";

pub fn main() {
    println!("Loading Safetensors model weights from file: {WEIGHTS_FILE}");
    // Load Safetensors weights exported from PyTorch into a model record.
    let record: ModelRecord<B> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(WEIGHTS_FILE.into(), &Default::default())
        .expect("Failed to load Safetensors model weights");

    // Infer using the loaded model record.
    infer(record);
}
