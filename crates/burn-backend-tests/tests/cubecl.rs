//! CubeCL kernel tests.

#![recursion_limit = "256"]
// TODO: some tests are failing WITH fusion. Should validate the existing failures.
#[cfg(all(feature = "cube", not(feature = "fusion")))]
#[path = "."]
mod cube {
    type FloatElem = f32;
    type IntElem = i32;

    mod backend {
        include!("common/backend.rs");

        pub struct ReferenceDevice;

        impl ReferenceDevice {
            pub fn new() -> burn_dispatch::DispatchDevice {
                burn_dispatch::DispatchDevice::NdArray(Default::default())
            }
        }
    }
    pub use backend::*;

    #[path = "cubecl/mod.rs"]
    mod kernel;
}
