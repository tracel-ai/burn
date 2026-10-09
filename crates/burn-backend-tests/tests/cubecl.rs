//! CubeCL kernel tests.
#![cfg(feature = "cube")]

#[path = "."]
mod cube {
    type FloatElem = f32;
    type IntElem = i32;

    mod backend {
        include!("common/backend.rs");

        pub struct ReferenceDevice;

        impl ReferenceDevice {
            // Flex keeps sub-byte reference values in native i8 storage.
            pub fn new() -> burn_tensor::Device {
                burn_tensor::Device::flex()
            }
        }
    }
    pub use backend::*;

    #[path = "cubecl/mod.rs"]
    mod kernel;
}
