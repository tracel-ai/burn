//! Shared fixtures for the simd kernel tests.

use burn_backend::{Element, ElementConversion};
use ndarray::{Array4, ArrayD, IxDyn, ShapeBuilder};

use crate::SharedArray;

/// Flat standard-layout array. Lengths above ~64 exercise the blocked,
/// lane-sized, and scalar-tail loops on any lane count.
pub fn arr<T: Clone>(data: Vec<T>) -> SharedArray<T> {
    let len = data.len();
    ArrayD::from_shape_vec(IxDyn(&[len]), data)
        .unwrap()
        .into_shared()
}

/// Standard-layout array of a given shape.
pub fn arr_at<T: Clone>(shape: &[usize], data: Vec<T>) -> SharedArray<T> {
    ArrayD::from_shape_vec(IxDyn(shape), data)
        .unwrap()
        .into_shared()
}

/// `[N, C, H, W]` tensor backed by channels-last storage, so
/// `strides()[1] == 1` as the simd pool kernels require.
pub fn nhwc<E: Element + ElementConversion>(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> SharedArray<E> {
    Array4::from_shape_vec((n, h, w, c), vals(n * h * w * c))
        .unwrap()
        .permuted_axes([0, 3, 1, 2])
        .to_shared()
        .into_dyn()
}

/// Standard-layout `[N, C, H, W]` tensor.
pub fn nchw<E: Element + ElementConversion>(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> SharedArray<E> {
    arr_at(&[n, c, h, w], vals(n * c * h * w))
}

/// Deterministic non-negative pattern, safe for unsigned dtypes.
pub fn vals<E: ElementConversion>(n: usize) -> Vec<E> {
    (0..n).map(|i| E::from_elem(i % 11)).collect()
}

/// A second pattern with a different phase than `vals`, so elementwise ops
/// actually compare differing operands.
pub fn vals_b<E: ElementConversion>(n: usize) -> Vec<E> {
    (0..n).map(|i| E::from_elem((i * 7) % 11)).collect()
}

/// Deterministic strictly-positive pattern, for divisors.
pub fn pos<E: ElementConversion>(n: usize) -> Vec<E> {
    (0..n).map(|i| E::from_elem(i % 11 + 1)).collect()
}

/// Fortran-order 2-D array: contiguous but not standard layout.
pub fn f_order<T: Clone>(shape: [usize; 2], data: Vec<T>) -> SharedArray<T> {
    ArrayD::from_shape_vec(IxDyn(&shape).f(), data)
        .unwrap()
        .into_shared()
}

/// Flat array over stride-2 (gapped) storage, so no memory-order slice exists.
pub fn gapped<T: Clone>(data: Vec<T>) -> SharedArray<T> {
    let n = data.len() / 2;
    ArrayD::from_shape_vec(IxDyn(&[n]).strides(IxDyn(&[2])), data)
        .unwrap()
        .into_shared()
}

/// Unwrap a simd fast-path result, failing the test when it falls back.
pub fn simd<T, F>(result: Result<SharedArray<T>, F>) -> SharedArray<T> {
    match result {
        Ok(out) => out,
        Err(_) => panic!("simd path rejected the input"),
    }
}
