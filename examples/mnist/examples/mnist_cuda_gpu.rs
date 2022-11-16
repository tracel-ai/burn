use burn::tensor::backend::ADBackendDecorator;
use burn_tch::{TchBackend, TchDevice};
use mnist::training;

fn main() {
    let device = TchDevice::Cuda(0);
    training::run::<ADBackendDecorator<TchBackend<burn::tensor::f16>>>(device);
    println!("Done.");
}
