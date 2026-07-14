//! Burn autodiff tests.

#![recursion_limit = "256"]
#![cfg(any(
    feature = "vulkan",
    feature = "cuda",
    feature = "rocm",
    feature = "metal"
))]

extern crate alloc;

pub type FloatElem = burn_tensor::f16;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[allow(clippy::module_inception)]
#[path = "common/autodiff.rs"]
mod autodiff;
