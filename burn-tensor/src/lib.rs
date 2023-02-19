#![cfg_attr(not(test), no_std)]


#[macro_use]
extern crate derive_new;

mod tensor;

#[cfg(feature = "export_tests")]
mod tests;

pub use half::f16;
pub use tensor::*;
