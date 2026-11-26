use super::*;
use burn_tensor::{TensorData, Tolerance};

// Regression for https://github.com/tracel-ai/burn/issues/5122: fused autotune
// could select a VecMat-only kernel for MatVec and leave every row after zero unwritten.
#[test]
fn matvec_then_add_should_compute_every_row() {
    test_stream().executes(|| {
        let device = Default::default();

        let lhs = TestTensor::<2>::from_floats(
            [
                [0.9, 0.8, -0.9],
                [0.1, 0.45, -0.6],
                [-0.25, -0.41, 0.51],
                [0.6, 0.1, 0.0],
            ],
            &device,
        );
        let rhs = TestTensor::<2>::from_floats([[0.3], [-0.2], [0.1]], &device);

        // Keep the trailing addition lazy so it can fuse with [4, 3] @ [3, 1].
        let out = lhs.matmul(rhs);
        (out.clone() + out)
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &TensorData::from([[0.04], [-0.24], [0.116], [0.32]]),
                Tolerance::default().set_half_precision_absolute(1e-3),
            );
    });
}

// A tail adding a tensor the product owns may write its output in place of that tensor; a fused
// matmul whose output aliased it left the buffer holding the addend. The first call races the
// candidates on forked contexts, which never alias, so the later calls run the tuned winner.
#[test]
fn matmul_then_add_owned_tensor_should_write_the_sum() {
    test_stream().executes(|| {
        let device = Default::default();
        let [m, k, n] = [16, 32, 16];
        let value = |i: usize, modulo: usize| ((i % modulo) as f32 - modulo as f32 / 2.0) / 8.0;
        let lhs: Vec<f32> = (0..m * k).map(|i| value(i, 7)).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| value(i, 5)).collect();
        let addend: Vec<f32> = (0..m * n).map(|i| value(i, 3)).collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                let product: f32 = (0..k).map(|c| lhs[row * k + c] * rhs[c * n + col]).sum();
                product + addend[i]
            })
            .collect();
        let expected = TensorData::new(expected, [m, n]);

        let lhs = TestTensor::<1>::from_floats(lhs.as_slice(), &device).reshape([m, k]);
        let rhs = TestTensor::<1>::from_floats(rhs.as_slice(), &device).reshape([k, n]);
        for _ in 0..3 {
            let addend = TestTensor::<1>::from_floats(addend.as_slice(), &device).reshape([m, n]);
            (lhs.clone().matmul(rhs.clone()) + addend)
                .into_data()
                .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
        }
    });
}

// A projection that flattens its batches into rows, `x.reshape([rows, k]) · w`, then lays its
// product back in the batches for a residual add: the reshape of the product is the routine
// writing it reshaped, so the matmul and the add run as one fused matmul.
#[cfg(feature = "cube")]
#[test]
fn flattened_projection_fuses_its_residual_add() {
    use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};

    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let [batches, seq, k, n] = [2, 3, 32, 16];
        let value = |i: usize, modulo: usize| ((i % modulo) as f32 - modulo as f32 / 2.0) / 8.0;
        let x: Vec<f32> = (0..batches * seq * k).map(|i| value(i, 7)).collect();
        let w: Vec<f32> = (0..k * n).map(|i| value(i, 5)).collect();
        let residual: Vec<f32> = (0..batches * seq * n).map(|i| value(i, 3)).collect();
        let expected: Vec<f32> = (0..batches * seq * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                (0..k).map(|c| x[row * k + c] * w[c * n + col]).sum::<f32>() + residual[i]
            })
            .collect();
        let expected = TensorData::new(expected, [batches, seq, n]);

        let x = TestTensor::<1>::from_floats(x.as_slice(), &device).reshape([batches, seq, k]);
        let w = TestTensor::<1>::from_floats(w.as_slice(), &device).reshape([k, n]);
        let residual =
            TestTensor::<1>::from_floats(residual.as_slice(), &device).reshape([batches, seq, n]);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let product = x.reshape([batches * seq, k]).matmul(w);
        (product.reshape([batches, seq, n]) + residual)
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");
        let block = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .find(|block| block.operations.iter().any(matchers::is_matmul()))
            .unwrap_or_else(|| panic!("no block holds the matmul\n\n{tables}"));
        assert!(
            matches!(block.kind, BlockKind::Fused { name: "Matmul", .. }),
            "expected the matmul in a fused Matmul block, got {:?}\n\n{tables}",
            block.kind,
        );
        assert!(
            block.operations.iter().any(matchers::is_reshape()),
            "the product's reshape should share the matmul's block\n\n{tables}",
        );
    });
}

fn small_value(i: usize, modulo: usize) -> f32 {
    ((i % modulo) as f32 - modulo as f32 / 2.0) / 8.0
}

/// `lhs [1, k] · rhs [k, n]`, a row, in `f32`.
fn single_row_product(lhs: &[f32], rhs: &[f32], n: usize) -> Vec<f32> {
    (0..n)
        .map(|col| (0..lhs.len()).map(|c| lhs[c] * rhs[c * n + col]).sum())
        .collect()
}

// A single-row product whose shape no plain matmul ever launches: the fused autotune sees its
// key before the plain one is settled, then after. Every run must write the activated product,
// whichever candidate the race elects.
#[test]
fn single_row_matmul_then_silu_should_compute_across_elections() {
    test_stream().executes(|| {
        let device = Default::default();
        let [k, n] = [48, 40];
        let lhs: Vec<f32> = (0..k).map(|i| small_value(i, 7)).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| small_value(i, 5)).collect();
        let expected: Vec<f32> = single_row_product(&lhs, &rhs, n)
            .into_iter()
            .map(|x| x / (1.0 + (-x).exp()))
            .collect();
        let expected = TensorData::new(expected, [1, n]);

        for _ in 0..5 {
            let lhs = TestTensor::<1>::from_floats(lhs.as_slice(), &device).reshape([1, k]);
            let rhs = TestTensor::<1>::from_floats(rhs.as_slice(), &device).reshape([k, n]);
            burn_tensor::activation::silu(lhs.matmul(rhs))
                .into_data()
                .assert_approx_eq::<FloatElem>(
                    &expected,
                    Tolerance::default().set_half_precision_absolute(1e-2),
                );
        }
    });
}

// The same for a residual add, the other tail a projection fuses.
#[test]
fn single_row_matmul_then_residual_should_compute_across_elections() {
    test_stream().executes(|| {
        let device = Default::default();
        let [k, n] = [56, 24];
        let lhs: Vec<f32> = (0..k).map(|i| small_value(i, 7)).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| small_value(i, 5)).collect();
        let residual: Vec<f32> = (0..n).map(|i| small_value(i, 3)).collect();
        let expected: Vec<f32> = single_row_product(&lhs, &rhs, n)
            .into_iter()
            .zip(&residual)
            .map(|(x, r)| x + r)
            .collect();
        let expected = TensorData::new(expected, [1, n]);

        for _ in 0..5 {
            let lhs = TestTensor::<1>::from_floats(lhs.as_slice(), &device).reshape([1, k]);
            let rhs = TestTensor::<1>::from_floats(rhs.as_slice(), &device).reshape([k, n]);
            let residual =
                TestTensor::<1>::from_floats(residual.as_slice(), &device).reshape([1, n]);
            (lhs.matmul(rhs) + residual)
                .into_data()
                .assert_approx_eq::<FloatElem>(
                    &expected,
                    Tolerance::default().set_half_precision_absolute(1e-2),
                );
        }
    });
}
