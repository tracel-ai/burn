extern crate alloc;

#[cfg(test)]
type FloatElemType = f32;
#[cfg(test)]
type IntElemType = i32;

#[cfg(test)]
mod tests;

// TODO: tests should be split per-module for int / float so we can only re-run tests that are affected by dtype configuration

/// Generate a test module with custom floating & integer element types.
#[macro_export]
macro_rules! test_elem_variant {
    ($modname:ident, $float:ty, $int:ty, [$($feat:literal),* $(,)?]) => {
        // #[cfg(all(test, feature = "fusion"))]
        // #[cfg(test)]
        #[cfg(all(test, any($(feature = $feat),*)))]
        mod $modname {
            pub type FloatElemType = $float;
            pub type IntElemType = $int;

            mod tests {
                pub use super::*;

                include!("tests.rs");
            }
        }
    };
}

test_elem_variant!(
    f16_i32,
    burn_tensor::f16,
    i32,
    ["vulkan", "cuda", "rocm", "metal"]
);

// test_elem_variant!(bf16, burn_tensor::bf16,   i32);

// Don't test `flex32` for now, burn sees it as `f32` but is actually `f16` precision, so it
// breaks a lot of tests from precision issues
