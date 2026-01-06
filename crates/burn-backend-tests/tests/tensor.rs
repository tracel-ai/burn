//! Burn backend tensor tests.

#![allow(clippy::single_range_in_vec_init, reason = "false positive")]
extern crate alloc;

pub type FloatElemType = f32;
#[allow(unused)]
pub type IntElemType = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[path = "common/tensor.rs"]
mod tensor;
