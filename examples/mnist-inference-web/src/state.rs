use crate::model::Model;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};

#[cfg(feature = "wgpu")]
pub type Device = WgpuDevice;

#[cfg(all(feature = "flex", not(feature = "wgpu")))]
pub type Device = burn::backend::flex::FlexDevice;

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model {
    let device = Device::default();

    #[cfg(feature = "wgpu")]
    init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;

    let model = Model::new(&device.into());
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(STATE_ENCODED, &device.into())
        .expect("Failed to decode state");

    model.load_record(record)
}
