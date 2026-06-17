use crate::model::Model;
use burn::{
    module::Module,
    prelude::Device,
    store::ModuleRecord,
    tensor::Bytes,
};

// NOTE: regenerate this asset in the burnpack format (e.g. `model.into_record().save(..)`);
// the legacy bincode `model.bin` will not parse as burnpack at runtime.
static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model {
    #[cfg(all(feature = "flex", not(feature = "wgpu")))]
    let device = Device::flex();
    #[cfg(feature = "wgpu")]
    // Calls init_setup_async
    let device = Device::wgpu_async(Default::default()).await;

    let model = Model::new(&device);
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(STATE_ENCODED.to_vec()))
        .expect("Failed to decode state");

    model.load_record(record)
}
