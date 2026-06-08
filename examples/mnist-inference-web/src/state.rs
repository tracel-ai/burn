use crate::model::Model;
use burn::{
    module::Module,
    prelude::Device,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model {
    #[cfg(all(feature = "flex", not(feature = "wgpu")))]
    let device = Device::flex();
    #[cfg(feature = "wgpu")]
    // Calls init_setup_async
    let device = Device::wgpu_async(Default::default()).await;

    let model = Model::new(&device);
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(STATE_ENCODED, &device)
        .expect("Failed to decode state");

    model.load_record(record)
}
