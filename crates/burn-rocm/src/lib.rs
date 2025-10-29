#![cfg_attr(docsrs, feature(doc_auto_cfg))]

// On macOS (and other non-ROCm hosts) we provide a lightweight stub so the crate
// can be compiled in the workspace without requiring HIP libraries. The real
// implementation is exposed on supported platforms.

#[cfg(target_os = "macos")]
mod platform_stub {
    // Minimal stubs to satisfy consumers when ROCm is not available.
    #[allow(dead_code)]
    pub struct RocmDevicePlaceholder;

    use std::marker::PhantomData;

    /// Generic dummy Rocm struct used when ROCm is not available.
    /// The PhantomData ensures the generic parameters are referenced so
    /// the compiler does not complain about unused type parameters.
    pub struct Rocm<F = f32, I = i32, B = u8>(PhantomData<(F, I, B)>);

    impl<F, I, B> Default for Rocm<F, I, B> {
        fn default() -> Self {
            Rocm(PhantomData)
        }
    }
}

#[cfg(target_os = "macos")]
pub use platform_stub::Rocm as Rocm;

#[cfg(not(target_os = "macos"))]
extern crate alloc;

#[cfg(not(target_os = "macos"))]
use burn_cubecl::CubeBackend;

#[cfg(not(target_os = "macos"))]
pub use cubecl::hip::AmdDevice as RocmDevice;

#[cfg(not(target_os = "macos"))]
use cubecl::hip::HipRuntime;

#[cfg(all(not(target_os = "macos"), not(feature = "fusion")))]
pub type Rocm<F = f32, I = i32, B = u8> = CubeBackend<HipRuntime, F, I, B>;

#[cfg(all(not(target_os = "macos"), feature = "fusion"))]
pub type Rocm<F = f32, I = i32, B = u8> = burn_fusion::Fusion<CubeBackend<HipRuntime, F, I, B>>;

// Only run the HIP-specific tests on supported platforms
#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use burn_cubecl::CubeBackend;

    pub type TestRuntime = cubecl::hip::HipRuntime;
    use half::f16;

    // TODO: Add tests for bf16
    // burn_cubecl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    burn_cubecl::testgen_all!([f16, f32], [i32], [u32]);
}
