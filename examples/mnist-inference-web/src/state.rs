use crate::model::Model;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};

#[cfg(feature = "wgpu")]
pub type Backend = Wgpu<f32, i32>;

#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type Backend = burn::backend::ndarray::NdArray<f32>;

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model<Backend> {
    #[cfg(feature = "wgpu")]
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    let model: Model<Backend> = Model::new(&Default::default());
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(STATE_ENCODED, &Default::default())
        .expect("Failed to decode state");

    model.load_record(record)
}
