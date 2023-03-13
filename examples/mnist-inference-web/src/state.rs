use crate::model::Model;

use burn::module::Module;
use burn::module::State;
use burn_ndarray::NdArrayBackend;

use bincode::{
    config::{self, Configuration},
    serde::decode_from_slice,
};

pub type Backend = NdArrayBackend<f32>;

const BINCODE_CONF: Configuration = config::standard();

// Bundled bincode serialized model state object
// see https://github.com/bincode-org/bincode and https://doc.rust-lang.org/std/macro.include_bytes.html
static STATE_ENCODED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mnist_model_state.bincode"));

/// Builds and loads trained parameters into the model.
pub fn build_and_load_model() -> Model<Backend> {
    let model: Model<Backend> = Model::new();

    // TODO: fix forward slash to make the paths work in windows
    let (state, _len): (State<f32>, usize) =
        decode_from_slice(STATE_ENCODED, BINCODE_CONF).expect("Failed to decode state");

    model
        .load(&state)
        .expect("State could not be loaded into model")
}
