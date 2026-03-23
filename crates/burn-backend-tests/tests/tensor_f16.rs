// //! Burn backend tensor tests.

#![allow(clippy::single_range_in_vec_init, reason = "false positive")]
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
#[path = "fused_ops/mod.rs"]
mod fusion;
