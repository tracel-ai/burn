use crate::model::Model;
use burn::module::Module;
use burn::record::BinBytesRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;

use burn::backend::wgpu::{compute::init_async, AutoGraphicsApi, WgpuBackend, WgpuDevice};

pub type Backend = WgpuBackend<AutoGraphicsApi, f32, i32>;

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model<Backend> {
    init_async::<AutoGraphicsApi>(&WgpuDevice::default()).await;

    let model: Model<Backend> = Model::new();
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec())
        .expect("Failed to decode state");

    model.load_record(record)
}
