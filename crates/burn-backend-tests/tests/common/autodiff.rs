// Burn autodiff tests, reusable with element types.

pub use super::*;

// Autodiff-enabled device used for tests.
pub struct AutodiffDevice;

impl AutodiffDevice {
    pub fn new() -> burn_dispatch::DispatchDevice {
        let device = burn_dispatch::DispatchDevice::default();

        burn_dispatch::DispatchDevice::autodiff(device)
    }
}

#[path = "../autodiff/mod.rs"]
mod base;

mod checkpointing {
    pub use super::FloatElem;

    // TODO: should be enabled via device!

    // Override autodiff device
    pub struct AutodiffDevice;

    impl AutodiffDevice {
        pub fn new() -> burn_dispatch::DispatchDevice {
            let device = burn_dispatch::DispatchDevice::default();

            burn_dispatch::DispatchDevice::autodiff_checkpointed(
                device,
                burn_dispatch::CheckpointingStrategy::Balanced,
            )
        }
    }

    include!("../autodiff/mod.rs");
}

// use burn_backend_tests::test_float_elem_variant;

// // NOTE: this currently doesn't test checkpointing with different dtypes
// test_float_elem_variant!(
//     f16,
//     burn_tensor::f16,
//     "../autodiff/mod.rs",
//     ["vulkan", "cuda", "rocm", "metal"]
// );

// TODO: bf16 not yet supported on any backend for full test suite
// test_float_elem_variant!(
//     bf16,
//     burn_tensor::bf16,
//     "../autodiff/mod.rs",
//     [] // ["cuda", "rocm"] TODO, ["vulkan"] only supports bf16 for matmul, metal/wgpu doesn't support bf16
// );
