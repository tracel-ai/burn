//! Regression test for fusion broadcast/reduce output vectorization (issue #5060).
//!
//! A fused broadcast/reduce trace could register an owned output with a `vector_size` wider than
//! the write width of the block that produces it, emitting an invalid scalar-into-vector store that
//! corrupts adjacent lanes. A softmax over the last dimension is the minimal trigger. The bug only
//! manifests on backends that pick a wide output vectorization, so this is a lightweight guard.

use super::*;
use burn_tensor::{TensorData, Tolerance};

/// Reference softmax over the last dimension, computed in `f64` from the raw input values.
fn softmax_reference(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut expected = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row.iter().map(|&v| ((v - max) as f64).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for c in 0..cols {
            expected[r * cols + c] = (exps[c] / sum) as f32;
        }
    }
    expected
}

#[test]
fn test_reduce_broadcast_softmax_vectorized_output() {
    // `cols` is a multiple of common wide vector sizes (16/8/4/2) so the output gets a wide
    // vectorization while the reduce forces a narrower write width.
    let (rows, cols) = (64, 16);
    let device = Default::default();
    let n = (rows * cols) as i64;

    let x = TestTensorInt::<1>::arange(0..n, &device)
        .reshape([rows, cols])
        .float();

    // Force the input to be materialized so the softmax below forms a single fused trace.
    device.sync().unwrap();

    // Softmax over the last dim: reduce -> broadcast -> reduce -> broadcast.
    let max = x.clone().max_dim(1);
    let e = (x - max).exp();
    let denom = e.clone().sum_dim(1);
    let out = e / denom;

    let actual = out.into_data();

    let input: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let expected = TensorData::new(softmax_reference(&input, rows, cols), [rows, cols]);

    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
