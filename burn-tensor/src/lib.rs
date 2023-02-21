#![cfg_attr(not(any(feature = "std", test)), no_std)]

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

#[cfg(feature = "export_tests")]
mod tests;

pub use half::f16;
pub use tensor::*;
