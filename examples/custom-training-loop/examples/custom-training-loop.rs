use burn::backend::{Autodiff, WebGpu};

fn main() {
    custom_training_loop::run::<Autodiff<WebGpu>>(Default::default());
}
