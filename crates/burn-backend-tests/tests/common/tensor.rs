/// Burn backend tensor tests, reusable with element types.
pub use super::*;

#[path = "../tensor/clone_invariance.rs"]
mod clone_invariance;

#[path = "../tensor/multi_threads.rs"]
mod multi_threads;

// Default float dtype
#[path = "../tensor/float/mod.rs"]
mod float;

// Default integer dtype
#[path = "../tensor/int/mod.rs"]
mod int;

#[cfg(any(
    feature = "vulkan",
    feature = "cuda",
    feature = "rocm",
    feature = "metal"
))]
mod f16 {
    pub type FloatElemType = burn_tensor::f16;
    #[allow(unused)]
    pub use super::IntElemType;

    mod ty {
        include!("backend.rs");
        include!("../tensor/float/mod.rs");
    }
}

#[cfg(any(
    feature = "vulkan",
    // feature = "cuda", // TODO
    // feature = "rocm",
    feature = "metal"
))]
mod bf16 {
    pub type FloatElemType = burn_tensor::f16;
    #[allow(unused)]
    pub use super::IntElemType;

    mod ty {
        include!("backend.rs");
        include!("../tensor/float/mod.rs");
    }
}
