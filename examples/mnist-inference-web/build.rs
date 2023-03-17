use std::fs;

use burn::module::State;

use bincode::config;

const GENERATED_FILE_NAME: &str = "mnist_model_state.bincode";
const MODEL_STATE_FILE_NAME: &str = "model-4.json.gz";

/// This build step is responsible for converting JSON serialized to Bincode serilization
/// in order to make the file small and efficient for bundling the binary into wasm code.
///
/// This will be removed once https://github.com/burn-rs/burn/issues/201 is resolved.
fn main() {
    println!("cargo:rerun-if-changed={MODEL_STATE_FILE_NAME}");
    let config = config::standard();
    let path: std::path::PathBuf = [
        std::env::var("OUT_DIR").expect("No build target path set"),
        GENERATED_FILE_NAME.into(),
    ]
    .iter()
    .collect();
    let state: State<f32> =
        State::load(MODEL_STATE_FILE_NAME).expect(concat!("Model JSON file could not be loaded"));
    let serialized =
        bincode::serde::encode_to_vec(state, config).expect("Encoding state into bincode failed");
    fs::write(path, serialized).expect("Write failed");
}
