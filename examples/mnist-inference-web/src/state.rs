use crate::model::Model;

use burn::module::Module;
use burn::module::State;
use burn_ndarray::NdArrayBackend;

pub type Backend = NdArrayBackend<f32>;

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub fn build_and_load_model() -> Model<Backend> {
    let model: Model<Backend> = Model::new();
    let state: State<f32> = State::from_bin(STATE_ENCODED).expect("Failed to decode state");

    model
        .load(&state)
        .expect("State could not be loaded into model")
}
