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
            // NdArray stays the reference implementation while it is deprecated: its
            // `export_tests` feature widens the accepted quantization schemes (Q4/Q2) so it can
            // serve as a value-equality reference, which no other CPU backend currently does.
            #[allow(deprecated)]
            pub fn new() -> burn_tensor::Device {
                burn_ndarray::NdArrayDevice::Cpu.into()
            }
        }
    }
    pub use backend::*;

    #[path = "cubecl/mod.rs"]
    mod kernel;
}
