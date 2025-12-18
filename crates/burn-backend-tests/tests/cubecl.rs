//! CubeCL kernel tests.

#[cfg(feature = "cube")]
#[path = "."]
mod cube {
    type FloatElemType = f32;
    type IntElemType = i32;

    mod backend {
        include!("common/backend.rs");
        pub type ReferenceBackend = burn_ndarray::NdArray<FloatElemType>;
    }
    pub use backend::*;

    #[path = "cubecl/mod.rs"]
    mod kernel;
}
