use burn::backend::{Autodiff, Wgpu};

fn main() {
    custom_training_loop::run::<Autodiff<Wgpu>>(Default::default());
}
