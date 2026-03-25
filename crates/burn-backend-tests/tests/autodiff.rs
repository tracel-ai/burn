//! Burn autodiff tests.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init, clippy::duplicate_mod)]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[allow(clippy::module_inception)]
#[path = "common/autodiff.rs"]
mod autodiff;
