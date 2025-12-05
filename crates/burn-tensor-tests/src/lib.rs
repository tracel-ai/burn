extern crate alloc;

#[cfg(test)]
type FloatElemType = f32;
#[cfg(test)]
type IntElemType = i32;

#[cfg(test)]
mod tests;

/// Generate a test module with custom floating & integer element types.
#[macro_export]
macro_rules! test_elem_variant {
    ($modname:ident, $float:ty, $int:ty, [$($feat:literal),* $(,)?]) => {
        #[cfg(all(test, any($(feature = $feat),*)))]
        mod $modname {
            pub type FloatElemType = $float;
            pub type IntElemType = $int;

            mod tests {
                include!("tests.rs");
            }
        }
    };
}

test_elem_variant!(
    f16_i16,
    burn_tensor::f16,
    // TODO: tests should be split per-module for int / float so we can only re-run tests that are affected by dtype configuration
    i32,
    ["vulkan", "cuda", "rocm", "metal"]
);

// test_elem_variant!(bf16, burn_tensor::bf16,   i32);

// Don't test `flex32` for now, burn sees it as `f32` but is actually `f16` precision, so it
// breaks a lot of tests from precision issues

// #[cfg(not(target_os = "macos"))] // Wgpu on MacOS currently doesn't support atomic compare exchange
// burn_autodiff::testgen_ad_deform_conv2d!(); // This kernel in cubecl isn't implemented without atomics

// TODO: maybe remove q_* ops for equality?
// TODO: remove q remainder?

/*
CUDA
burn_cubecl::testgen_all!([f32], [i32], [u32]);

WGPU
mod tests {
    use burn_cubecl::CubeBackend;
    #[cfg(feature = "vulkan")]
    pub use half::f16;
    #[cfg(feature = "metal")]
    pub use half::f16;

    pub type TestRuntime = cubecl::wgpu::WgpuRuntime;


    #[cfg(feature = "vulkan")]
    burn_cubecl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    #[cfg(feature = "metal")]
    burn_cubecl::testgen_all!([f16, f32], [i16, i32], [u32]);
    #[cfg(all(not(feature = "vulkan"), not(feature = "metal")))]
    burn_cubecl::testgen_all!([f32], [i32], [u32]);
}

ROCM
burn_cubecl::testgen_all!([f16, f32], [i32], [u32]);

CPU
burn_cubecl::testgen_all!([f32], [i8, i16, i32, i64], [u32]);
*/
