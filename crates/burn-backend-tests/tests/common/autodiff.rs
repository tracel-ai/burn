// Burn autodiff tests, reusable with element types.

pub use super::*;

// Autodiff-enabled device used for tests.
pub struct AutodiffDevice;

impl AutodiffDevice {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> burn_tensor::Device {
        burn_tensor::Device::default().autodiff()
    }
}

#[path = "../autodiff/mod.rs"]
mod base;

mod checkpointing {
    pub use super::FloatElem;

    // Override autodiff device
    pub struct AutodiffDevice;

    impl AutodiffDevice {
        #[allow(clippy::new_ret_no_self)]
        pub fn new() -> burn_tensor::Device {
            burn_tensor::Device::default()
                .autodiff()
                .gradient_checkpointing()
        }
    }

    include!("../autodiff/mod.rs");
}
