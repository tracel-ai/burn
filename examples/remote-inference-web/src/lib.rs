#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod model;

#[cfg(target_family = "wasm")]
pub mod web;
