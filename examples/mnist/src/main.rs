pub mod data;
pub mod model;

mod training;

fn main() {
    use burn::tensor::backend::{NdArrayADBackend, NdArrayDevice};

    let device = NdArrayDevice::Cpu;
    training::run::<NdArrayADBackend<f32>>(device);
    println!("Done.");
}
