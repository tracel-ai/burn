//! Minimal native-kernel crate with a Rust GEMM fallback and an FFI shim for a Zig-built GEMM.
//!
//! The Zig kernel is feature-gated behind the `zig` feature. By default the pure-Rust
//! implementation is used so CI and macOS builds don't require Zig or native toolchains.

#[cfg(feature = "zig")]
extern "C" {
    // C ABI: gemm_f32(a_ptr, b_ptr, c_ptr, m, n, k)
    // All pointers are row-major f32 slices.
    fn gemm_f32(a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize);
    // Quantized u8 GEMM fast path: produce f32 accumulators from u8 inputs.
    // Rust side will requantize the f32 outputs to u8.
    fn gemm_q8_to_i64(a: *const u8, b: *const u8, c: *mut i64, m: usize, n: usize, k: usize);
}

/// Pure Rust reference quantized GEMM: dequantize -> gemm_rust -> requantize
pub fn gemm_q8_rust(a: &[u8], b: &[u8], c: &mut [u8], m: usize, n: usize, k: usize, scale_a: f32, scale_b: f32, scale_out: f32) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    // allocate temporary f32 buffers
    let mut a_f = vec![0f32; m * k];
    let mut b_f = vec![0f32; k * n];
    let mut c_f = vec![0f32; m * n];

    for i in 0..(m * k) {
        a_f[i] = (a[i] as f32) * scale_a;
    }
    for i in 0..(k * n) {
        b_f[i] = (b[i] as f32) * scale_b;
    }

    gemm_rust(&a_f, &b_f, &mut c_f, m, n, k);

    for i in 0..(m * n) {
        let q = (c_f[i] / scale_out).round();
        let q = if q < 0.0 { 0.0 } else if q > 255.0 { 255.0 } else { q };
        c[i] = q as u8;
    }
}

/// Pure Rust reference GEMM (row-major): C = A * B
pub fn gemm_rust(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Call the Zig/native kernel when the feature is enabled, otherwise fall back to Rust.
pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(feature = "zig")]
    unsafe {
        gemm_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k);
    }

    #[cfg(not(feature = "zig"))]
    {
        gemm_rust(a, b, c, m, n, k);
    }
}

/// Call quantized kernel when available; otherwise fallback to Rust dequantize path.
pub fn gemm_q8(a: &[u8], b: &[u8], c: &mut [u8], m: usize, n: usize, k: usize, scale_a: f32, scale_b: f32, scale_out: f32) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(feature = "zig")]
    unsafe {
        // Call Zig to compute integer accumulators into a temporary i64 buffer, then apply scales and requantize in Rust.
        let mut tmp = vec![0i64; m * n];
        gemm_q8_to_i64(a.as_ptr(), b.as_ptr(), tmp.as_mut_ptr(), m, n, k);
        let scale = scale_a * scale_b;
        for i in 0..(m * n) {
            let sumf = (tmp[i] as f32) * scale;
            let q = (sumf / scale_out).round();
            let q = if q < 0.0 { 0.0 } else if q > 255.0 { 255.0 } else { q };
            c[i] = q as u8;
        }
    }

    #[cfg(not(feature = "zig"))]
    {
        gemm_q8_rust(a, b, c, m, n, k, scale_a, scale_b, scale_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn rand_data(len: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            v.push(((i * 31 + 7) % 101) as f32 * 0.013f32);
        }
        v
    }

    #[test]
    fn gemm_small() {
        let m = 5;
        let n = 7;
        let k = 3;
        let a = rand_data(m * k);
        let b = rand_data(k * n);
        let mut c_ref = vec![0f32; m * n];
        let mut c_tgt = vec![0f32; m * n];

        gemm_rust(&a, &b, &mut c_ref, m, n, k);
        gemm(&a, &b, &mut c_tgt, m, n, k);

        for (r, t) in c_ref.iter().zip(c_tgt.iter()) {
            assert_abs_diff_eq!(r, t, epsilon = 1e-6f32);
        }
    }
}
