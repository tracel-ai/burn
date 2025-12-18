//! Burn autodiff tests.

#![allow(
    clippy::single_range_in_vec_init,
    clippy::duplicate_mod,
    reason = "false positive"
)]
extern crate alloc;

pub type FloatElemType = f32;
#[allow(unused)]
pub type IntElemType = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[allow(clippy::module_inception)]
#[path = "common/autodiff.rs"]
mod autodiff;
