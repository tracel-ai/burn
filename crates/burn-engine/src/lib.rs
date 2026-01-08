#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]

//! Burn multi-backend engine.

#[cfg(not(any(
    feature = "cpu",
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    feature = "webgpu",
)))]
compile_error!("At least one backend feature must be enabled.");

#[cfg(any(
    all(feature = "vulkan", feature = "metal"),
    all(feature = "vulkan", feature = "webgpu"),
    all(feature = "metal", feature = "webgpu")
))]
compile_error!("Only one wgpu runtime feature can be enabled at once.");

mod engine;

pub use engine::*;

extern crate alloc;
