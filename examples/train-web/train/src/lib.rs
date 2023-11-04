use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use wasm_bindgen::prelude::*;

mod train;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init().expect("Error initializing logger");
}

#[wasm_bindgen]
pub fn run() {
    log::info!("Hello from Rust");
    train::run::<Autodiff<NdArray<f32>>>(NdArrayDevice::Cpu);
}
