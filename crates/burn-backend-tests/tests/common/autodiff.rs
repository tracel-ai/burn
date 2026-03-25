// Burn autodiff tests, reusable with element types.

pub use super::*;

// Autodiff-enabled device used for tests.
pub struct AutodiffDevice;

impl AutodiffDevice {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> burn_dispatch::DispatchDevice {
        let device = burn_dispatch::DispatchDevice::default();

        burn_dispatch::DispatchDevice::autodiff(device)
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
        pub fn new() -> burn_dispatch::DispatchDevice {
            let device = burn_dispatch::DispatchDevice::default();

            burn_dispatch::DispatchDevice::autodiff_checkpointed(
                device,
                burn_dispatch::CheckpointingStrategy::Balanced,
            )
        }
    }

    include!("../autodiff/mod.rs");
}
