// //! Burn backend tensor tests.

#![recursion_limit = "256"]
#![cfg(any(
    feature = "vulkan",
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "flex"
))]

extern crate alloc;

pub type FloatElem = burn_tensor::f16;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[path = "tensor/float/mod.rs"]
mod f16;

#[cfg(feature = "fusion")]
#[path = "fusion/mod.rs"]
mod fusion;

// TODO: bf16 (vulkan only supports bf16 for matmul, metal/wgpu doesn't support bf16)
