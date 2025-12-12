extern crate alloc;

#[cfg(feature = "std")]
pub use burn_tensor_testgen::might_panic;

#[allow(unused)]
type FloatElemType = f32;
#[allow(unused)]
type IntElemType = i32;

#[cfg(test)]
mod backend;
#[cfg(test)]
pub use backend::*;

#[cfg(test)]
mod tests;

/// CubeCL kernel tests.
#[cfg(all(test, feature = "cube"))]
#[path = "."]
mod cube {
    type FloatElemType = f32;
    type IntElemType = i32;

    mod backend {
        include!("backend.rs");
        pub type ReferenceBackend = burn_ndarray::NdArray<FloatElemType>;
    }
    pub use backend::*;

    #[path = "cubecl/mod.rs"]
    mod kernel;
}

/// Generate a test module with custom floating & integer element types.
#[macro_export]
macro_rules! test_elem_variant {
    ($modname:ident, $float:ty, $int:ty, [$($feat:literal),* $(,)?]) => {
        #[cfg(all(test, any($(feature = $feat),*)))]
        mod $modname {
            pub type FloatElemType = $float;
            pub type IntElemType = $int;

            mod tests {
                include!("backend.rs");
                include!("tests.rs");
            }
        }
    };
}

test_elem_variant!(
    f16_i32,
    burn_tensor::f16,
    // TODO: tests should be split per-module for int / float so we can only re-run tests that are affected by dtype configuration
    i32,
    ["vulkan", "cuda", "rocm", "metal"]
);

// test_elem_variant!(bf16, burn_tensor::bf16,   i32);

// Don't test `flex32` for now, burn sees it as `f32` but is actually `f16` precision, so it
// breaks a lot of tests from precision issues
