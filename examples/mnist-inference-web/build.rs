use burn::module::{State, StateFormat};

const GENERATED_FILE_NAME: &str = "mnist_model_state";
const MODEL_STATE_FILE_NAME: &str = "model-4";

/// This build step is responsible for converting JSON serialized to Bincode serilization
/// in order to make the file small and efficient for bundling the binary into wasm code.
///
/// This will be removed once https://github.com/burn-rs/burn/issues/201 is resolved.
fn main() {
    println!("cargo:rerun-if-changed={MODEL_STATE_FILE_NAME}");
    let path: std::path::PathBuf = [
        std::env::var("OUT_DIR").expect("No build target path set"),
        GENERATED_FILE_NAME.into(),
    ]
    .iter()
    .collect();
    let state: State<f32> = State::load(MODEL_STATE_FILE_NAME, &StateFormat::JsonGz)
        .expect(concat!("Model JSON file could not be loaded"));
    state
        .save_bin(path.to_str().expect("Valid path"))
        .expect("Write failed");
}
