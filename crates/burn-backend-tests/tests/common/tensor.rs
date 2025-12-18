#![allow(clippy::single_range_in_vec_init, reason = "false positive")]

/// Burn backend tensor tests, reusable with element types.
pub use super::*;

mod backend;
pub use backend::*;

#[path = "../tensor/mod.rs"]
mod tensor;
