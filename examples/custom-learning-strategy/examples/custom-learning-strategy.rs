use burn::backend::{Autodiff, WebGpu};

fn main() {
    custom_learning_strategy::training::run::<Autodiff<WebGpu>>(Default::default());
}
