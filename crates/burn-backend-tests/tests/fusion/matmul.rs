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
                [1.0, 0.75, -1.0],
                [0.125, 0.5, -0.625],
                [-0.25, -0.5, 0.5],
                [0.5, 0.125, 0.0],
            ],
            &device,
        );
        // Binary fractions keep the row-coverage check independent of TF32 rounding.
        let rhs = TestTensor::<2>::from_floats([[0.25], [-0.125], [0.125]], &device);

        // Keep the trailing addition lazy so it can fuse with [4, 3] @ [3, 1].
        let out = lhs.matmul(rhs);
        (out.clone() + out)
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &TensorData::from([[0.0625], [-0.21875], [0.125], [0.21875]]),
                Tolerance::default().set_half_precision_absolute(1e-3),
            );
    });
}
