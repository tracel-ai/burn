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
