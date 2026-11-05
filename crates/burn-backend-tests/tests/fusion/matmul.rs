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
