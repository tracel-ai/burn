//! Burn backend tensor tests.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init)]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[path = "common/tensor.rs"]
mod tensor;

#[cfg(all(feature = "cube", feature = "fusion"))]
#[path = "fusion/mod.rs"]
mod fusion;
