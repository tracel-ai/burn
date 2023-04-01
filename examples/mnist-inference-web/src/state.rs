use crate::model::Model;

use burn::module::Module;
use burn::record::InMemoryBinRecorder;
use burn::record::Record;
use burn::record::Settings;
use burn_ndarray::NdArrayBackend;

pub type Backend = NdArrayBackend<f32>;

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub fn build_and_load_model() -> Model<Backend> {
    let model: Model<Backend> = Model::new();
    let state = Record::load::<Settings<f32, f32, InMemoryBinRecorder>>(STATE_ENCODED.to_vec())
        .expect("Failed to decode state");

    model
        .load(&state)
        .expect("State could not be loaded into model")
}
