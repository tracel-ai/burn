/// Whether SIMD instructions are worth using
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
pub fn should_use_simd(len: usize) -> bool {
    len >= 128
}

/// Whether SIMD instructions are worth using
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
pub fn should_use_simd(_len: usize) -> bool {
    false
}
