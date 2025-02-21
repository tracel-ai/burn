#![no_std]
extern crate alloc;
mod util;

pub mod proto {
    pub const MNIST_IMAGE_HEIGHT: usize = 28;
    pub const MNIST_IMAGE_WIDTH: usize = 28;
    pub const MNIST_IMAGE_SIZE: usize = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT;
    pub type MnistImage = [u8; MNIST_IMAGE_SIZE];

    pub struct Output {
        pub loss: f32,
        pub accuracy: f32,
    }
}
pub mod inference;
mod model;
pub mod train;
