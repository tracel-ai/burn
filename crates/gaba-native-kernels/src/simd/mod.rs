// SIMD support scaffolding for gaba-native-kernels.
// Provides a dispatching `gemm_simd` function that uses architecture-specific
// implementations when available (AVX2 on x86_64, NEON on aarch64).

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon;

/// Try a SIMD GEMM implementation. If SIMD feature is not enabled or no arch-specific
/// implementation is provided, fall back to returning false so the caller can use the
/// portable Rust implementation.
pub fn gemm_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        return avx2::gemm_avx2(a, b, c, m, n, k);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return neon::gemm_neon(a, b, c, m, n, k);
    }

    // Not implemented for this platform or feature not enabled
    false
}
