use crate::model::Model;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{init_async, AutoGraphicsApi, Wgpu, WgpuDevice};

#[cfg(feature = "wgpu")]
pub type Backend = Wgpu<f32, i32>;

#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type Backend = burn::backend::ndarray::NdArray<f32>;

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model<Backend> {
    #[cfg(feature = "wgpu")]
    init_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    let model: Model<Backend> = Model::new(&Default::default());
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &Default::default())
        .expect("Failed to decode state");

    model.load_record(record)
}
