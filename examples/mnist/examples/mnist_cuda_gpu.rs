use mnist::training;

fn main() {
    use burn::tensor::backend::{TchADBackend, TchDevice};

    let device = TchDevice::Cuda(0);
    training::run::<TchADBackend<burn::tensor::f16>>(device);
    println!("Done.");
}
