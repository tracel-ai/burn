#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

#[cfg(feature = "export_tests")]
mod tests;

pub use half::{bf16, f16};
pub use tensor::*;
