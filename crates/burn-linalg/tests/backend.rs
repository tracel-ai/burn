#![cfg(any(
    feature = "flex",
    feature = "wgpu",
    feature = "webgpu",
    feature = "vulkan",
    feature = "wgpu-msl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "cpu",
    feature = "remote",
    feature = "capture"
))]

#[path = "backend/mod.rs"]
mod backend_tests;
