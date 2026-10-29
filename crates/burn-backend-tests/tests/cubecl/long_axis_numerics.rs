//! The chunked routes (a sum or mean of few outputs over a long axis, a long-k matmul) against an f64
//! reference, with explicit error bounds: they add in another order than one chain per output, so
//! their values are checked, not their bits.
//!
//! A two-level sum of `n` terms in chunks of `CHUNK` has a worst-case error of
//! `(CHUNK + n / CHUNK) * eps * sum |x|` (one chain per chunk, then one over the chunks); the
//! bounds below are that one, rounded up.

use super::*;
use burn_tensor::{Device, TensorData};

const CHUNK: usize = 4096;

/// Uniform values in `[-1, 1)`, reproducible.
fn uniform(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The two-level sum's error bound for `n` terms of absolute sum `abs`.
fn bound(n: usize, abs: f64) -> f64 {
    ((CHUNK + n.div_ceil(CHUNK) + 1) as f64) * f64::from(f32::EPSILON) * abs
}

fn values(tensor: Tensor<2>) -> Vec<f32> {
    tensor.into_data().to_vec().unwrap()
}

#[track_caller]
fn assert_within(actual: f32, exact: f64, abs: f64, n: usize) {
    let error = (f64::from(actual) - exact).abs();
    assert!(
        error <= bound(n, abs),
        "{actual} is {error} from {exact}, past the bound {}",
        bound(n, abs)
    );
}

#[test]
fn a_chunked_sum_of_cancelling_terms_stays_within_its_bound() {
    // Terms of ±1e6 that cancel pairwise, plus a small remainder each.
    let n = 200_001;
    let small = uniform(2 * n, 1);
    let rows: Vec<f32> = (0..2 * n)
        .map(|i| if i % 2 == 0 { 1.0e6 } else { -1.0e6 } + small[i])
        .collect();
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(TensorData::new(rows.clone(), [2, n]), &device);
    let sums = values(tensor.clone().sum_dim(1));
    let means = values(tensor.mean_dim(1));
    for row in 0..2 {
        let terms = &rows[row * n..(row + 1) * n];
        let exact: f64 = terms.iter().map(|&x| f64::from(x)).sum();
        let abs: f64 = terms.iter().map(|&x| f64::from(x).abs()).sum();
        assert_within(sums[row], exact, abs, n);
        // The worst case is loose here (abs is 1e11 for a sum near 1e6): the actual error is far below it.
        assert!(
            (f64::from(sums[row]) - exact).abs() <= 1e-3 * exact.abs(),
            "{} for {exact}",
            sums[row]
        );
        assert_within(means[row] * n as f32, exact, abs * 2.0, n);
    }
}

#[test]
fn a_chunked_sum_over_a_strided_axis_stays_within_its_bound() {
    // The reduced axis has stride 3: the transpose of a contiguous [3, n].
    let n = 150_000;
    let data = uniform(3 * n, 2);
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(TensorData::new(data.clone(), [n, 3]), &device);
    let sums = values(tensor.sum_dim(0));
    for column in 0..3 {
        let terms = (0..n).map(|i| f64::from(data[i * 3 + column]));
        let exact: f64 = terms.clone().sum();
        let abs: f64 = terms.map(f64::abs).sum();
        assert_within(sums[column], exact, abs, n);
    }
}

#[test]
fn a_chunked_sum_propagates_overflow_infinities_and_nan() {
    let n = 70_000;
    let mut rows = vec![0.0f32; 4 * n];
    // Two terms near f32::MAX in different chunks: their sum overflows whatever the order.
    rows[10] = 3.0e38;
    rows[n - 10] = 3.0e38;
    rows[n + 12_345] = f32::NAN;
    rows[2 * n + 5] = f32::INFINITY;
    rows[3 * n + 7] = f32::INFINITY;
    rows[4 * n - 7] = f32::NEG_INFINITY;
    let device = Default::default();
    let sums =
        values(TestTensor::<2>::from_data(TensorData::new(rows, [4, n]), &device).sum_dim(1));
    assert_eq!(sums[0], f32::INFINITY);
    assert!(sums[1].is_nan());
    assert_eq!(sums[2], f32::INFINITY);
    assert!(sums[3].is_nan());
}

#[test]
fn a_chunked_sum_of_subnormals_is_exact_or_flushed() {
    // Sums of subnormals are exact in f32; a device that flushes subnormals to zero gives 0.
    let n = 70_000;
    let term = f32::MIN_POSITIVE / 4.0;
    let device = Default::default();
    let sum = values(
        TestTensor::<2>::from_data(TensorData::new(vec![term; n], [1, n]), &device).sum_dim(1),
    )[0];
    let exact = f64::from(term) * n as f64;
    assert!(
        sum == 0.0 || (f64::from(sum) - exact).abs() <= bound(n, exact),
        "{sum} for {exact}"
    );
}

#[test]
fn a_chunked_long_k_matmul_stays_within_its_bound() {
    let [m, n, k] = [3, 2, 100_000];
    let lhs = uniform(m * k, 3);
    let rhs = uniform(k * n, 4);
    let device = Default::default();
    let output = values(
        TestTensor::<2>::from_data(TensorData::new(lhs.clone(), [m, k]), &device).matmul(
            TestTensor::<2>::from_data(TensorData::new(rhs.clone(), [k, n]), &device),
        ),
    );
    for i in 0..m {
        for j in 0..n {
            let products = (0..k).map(|p| f64::from(lhs[i * k + p]) * f64::from(rhs[p * n + j]));
            let exact: f64 = products.clone().sum();
            let abs: f64 = products.map(f64::abs).sum();
            // Each product is rounded once more than the f64 one.
            assert_within(output[i * n + j], exact, abs * 2.0, k);
        }
    }
}

#[test]
fn the_gradients_through_the_chunked_routes_match_the_reference() {
    // d(sum_k sum_dim(x * w))/dx = w, and d(sum(a @ b))/da = rows of b summed over its columns.
    let [m, n, k] = [3, 2, 100_000];
    let device = Device::default().autodiff();
    let w = uniform(m * k, 5);
    let x = TestTensor::<2>::from_data(TensorData::new(uniform(m * k, 6), [m, k]), &device)
        .require_grad();
    let weights = TestTensor::<2>::from_data(TensorData::new(w.clone(), [m, k]), &device);
    let gradients = (x.clone() * weights).sum_dim(1).sum().backward();
    let grad: Vec<f32> = x.grad(&gradients).unwrap().into_data().to_vec().unwrap();
    assert_eq!(grad, w);

    let b = uniform(k * n, 7);
    let a = TestTensor::<2>::from_data(TensorData::new(uniform(m * k, 8), [m, k]), &device)
        .require_grad();
    let rhs = TestTensor::<2>::from_data(TensorData::new(b.clone(), [k, n]), &device);
    let gradients = a.clone().matmul(rhs).sum().backward();
    let grad: Vec<f32> = a.grad(&gradients).unwrap().into_data().to_vec().unwrap();
    for i in 0..m {
        for p in 0..k {
            let exact: f64 = (0..n).map(|j| f64::from(b[p * n + j])).sum();
            let error = (f64::from(grad[i * k + p]) - exact).abs();
            assert!(
                error <= 4.0 * f64::from(f32::EPSILON) * exact.abs().max(1e-30),
                "{error}"
            );
        }
    }
}
