// Burn backend tensor tests, reusable with element types.

pub use super::*;

#[path = "../tensor/clone_invariance.rs"]
mod clone_invariance;

#[cfg(feature = "std")]
#[path = "../tensor/multi_threads.rs"]
mod multi_threads;

// Default float dtype
#[path = "../tensor/float/mod.rs"]
mod float;

// Default integer dtype
#[path = "../tensor/int/mod.rs"]
mod int;

// Default bool dtype
#[path = "../tensor/bool/mod.rs"]
mod bool;

use burn_backend_tests::test_float_elem_variant;

test_float_elem_variant!(
    f16,
    burn_tensor::f16,
    "../tensor/float/mod.rs",
    ["vulkan", "cuda", "rocm", "metal"]
);

test_float_elem_variant!(
    bf16,
    burn_tensor::bf16,
    "../tensor/float/mod.rs",
    ["metal"] // ["cuda", "rocm"] TODO, ["vulkan"] only supports bf16 for matmul
);
