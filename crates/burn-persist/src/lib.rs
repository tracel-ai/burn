#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

#[macro_use]
extern crate alloc;

pub mod safetensors;

// Re-export commonly used types
pub use safetensors::*;
