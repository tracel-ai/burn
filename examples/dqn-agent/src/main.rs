use burn::backend::{
    autodiff::Autodiff,
    ndarray::{NdArray, NdArrayDevice},
};

fn main() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();

    // making a entry point for training.rs
    dqn_agent::training::run::<Backend>(device);
}
