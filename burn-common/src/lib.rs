#![cfg_attr(not(feature = "std"), no_std)]

pub mod id;

#[cfg(not(feature = "std"))]
pub mod rand;

pub mod stub;

extern crate alloc;
