use crate::{model::ModelConfig, train::TrainingConfig};
use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    optim::AdamConfig,
    train::util,
};
use wasm_bindgen::prelude::*;

mod data;
mod mnist;
mod model;
mod train;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init().expect("Error initializing logger");
}

#[wasm_bindgen]
pub fn init(worker_url: String) -> Result<(), String> {
    util::init(worker_url)
}

#[wasm_bindgen]
pub fn run(
    train_labels: &[u8],
    train_images: &[u8],
    train_lengths: &[u16],
    test_labels: &[u8],
    test_images: &[u8],
    test_lengths: &[u16],
) -> Result<(), String> {
    log::info!("Hello from Rust");
    let config = TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new());
    train::train::<Autodiff<NdArray<f32>>>(
        "",
        config,
        NdArrayDevice::Cpu,
        train_labels,
        train_images,
        train_lengths,
        test_labels,
        test_images,
        test_lengths,
    );
    Ok(())
}

#[wasm_bindgen]
pub fn child_entry_point(ptr: u32) {
    let work = unsafe { Box::from_raw(ptr as *mut Box<dyn FnOnce()>) };
    (*work)();
}
