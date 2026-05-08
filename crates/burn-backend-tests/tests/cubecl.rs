//! CubeCL kernel tests.
#![cfg(feature = "cube")]
#![recursion_limit = "256"]

#[path = "."]
mod cube {
    type FloatElem = f32;
    type IntElem = i32;

    mod backend {
        include!("common/backend.rs");

        pub struct ReferenceDevice;

        impl ReferenceDevice {
            pub fn new() -> burn_tensor::Device {
                burn_ndarray::NdArrayDevice::Cpu.into()
            }
        }
    }
    pub use backend::*;

    #[path = "cubecl/mod.rs"]
    mod kernel;
}
