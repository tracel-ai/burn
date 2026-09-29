//! A fused matmul whose element-wise epilogue reads a transposed view.
//!
//! The matmul loads a transposed operand along its contiguous dimension, the second to last.
//! When the epilogue reads the same operand, it shares the input's vector size but applies it
//! along the last dimension, reading the wrong elements. Muon's Newton-Schulz iteration
//! (`x * a + (b @ x)`, with `x` transposed for tall matrices) diverged to NaN this way.

use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_tensor::{TensorData, Tolerance};

/// Row-major `[rows, cols]` data, `rows * cols` distinct values.
fn data(rows: usize, cols: usize, offset: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (i as f32 * 0.37 + offset).sin())
        .collect()
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                out[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    out
}

/// `x * 2 + m @ x`, where `x` is a transposed view whose last use is the epilogue.
#[test]
fn matmul_epilogue_reads_transposed_view_last() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let (rows, cols) = (32, 4);
        let t_data = data(rows, cols, 0.0);
        let m_data = data(cols, cols, 1.0);
        let t = TestTensor::<2>::from_data(TensorData::new(t_data.clone(), [rows, cols]), &device);
        let m = TestTensor::<2>::from_data(TensorData::new(m_data.clone(), [cols, cols]), &device);

        let x_data = transpose(&t_data, rows, cols);
        let mx = matmul(&m_data, &x_data, cols, cols, rows);
        let expected: Vec<f32> = x_data.iter().zip(&mx).map(|(x, y)| x * 2.0 + y).collect();

        let inspector = FusionInspector::install(stream);
        let x = t.swap_dims(0, 1);
        let mx = m.matmul(x.clone());
        let out = x.mul_scalar(2.0).add(mx);
        let out = out.into_data();

        // Only meaningful if the epilogue is fused into the matmul.
        let reports = inspector.drain();
        assert!(
            reports
                .iter()
                .flat_map(|report| report.fused_blocks())
                .any(|block| block.operations.len() == 3),
            "the matmul and its epilogue should be one fused block:\n\n{}",
            reports
                .iter()
                .map(|report| report.format_table())
                .collect::<Vec<_>>()
                .join("\n\n"),
        );

        out.assert_approx_eq::<FloatElem>(
            &TensorData::new(expected, [cols, rows]),
            Tolerance::default(),
        );
    });
}
