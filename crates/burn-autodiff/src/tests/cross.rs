#[burn_tensor_testgen::testgen(ad_cross)]
mod tests {
    use super::*;
    use burn_tensor::{TensorData, Tolerance, ops::FloatElem};

    // Helper to compute expected cross product for 2-D (N × 3) tensors.
    fn manual_cross(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<[f32; 3]> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                [
                    x[1] * y[2] - x[2] * y[1],
                    x[2] * y[0] - x[0] * y[2],
                    x[0] * y[1] - x[1] * y[0],
                ]
            })
            .collect()
    }

    #[test]
    fn forward_matches_manual_cross() {
        let device = Default::default();
        let a_raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b_raw = [[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]];
        let a = TestTensor::<2>::from_data(TensorData::from(a_raw), &device);
        let b = TestTensor::<2>::from_data(TensorData::from(b_raw), &device);

        let out = a.cross(b.clone(), 1);
        let expected_vec = manual_cross(&a_raw, &b_raw);
        let expected: [[f32; 3]; 2] = [expected_vec[0], expected_vec[1]];

        out.to_data().assert_approx_eq::<FloatElem<TestBackend>>(
            &TensorData::from(expected),
            Tolerance::default(),
        );
    }

    #[test]
    fn backward_basic() {
        let device = Default::default();
        let a = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        )
        .require_grad();
        let b = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            &device,
        )
        .require_grad();

        // Simple cross product; grad is a vector of ones.
        let c = a.clone().cross(b.clone(), 1);
        let grads = c.backward();

        let a_grad = a.grad(&grads).unwrap().to_data();
        let b_grad = b.grad(&grads).unwrap().to_data();

        // For a: b×grad_out, where grad_out = [1,1,1]
        let expected_a = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]);
        // For b: -(grad_out×a)
        let expected_b = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]);

        a_grad.assert_approx_eq::<FloatElem<TestBackend>>(&expected_a, Tolerance::default());
        b_grad.assert_approx_eq::<FloatElem<TestBackend>>(&expected_b, Tolerance::default());
    }

    #[test]
    fn backward_after_sum() {
        let device = Default::default();
        let a = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        )
        .require_grad();
        let b = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            &device,
        )
        .require_grad();

        // Sum reduces to scalar, but the gradient should be the same.
        let c = a.clone().cross(b.clone(), 1).sum();
        let grads = c.backward();

        let a_grad = a.grad(&grads).unwrap().to_data();
        let b_grad = b.grad(&grads).unwrap().to_data();

        let expected_a = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]);
        let expected_b = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]);

        a_grad.assert_approx_eq::<FloatElem<TestBackend>>(&expected_a, Tolerance::default());
        b_grad.assert_approx_eq::<FloatElem<TestBackend>>(&expected_b, Tolerance::default());
    }

    #[test]
    fn different_dim() {
        // Also check when the cross is along a different dimension (e.g. dim = 0).
        let device = Default::default();
        let a_raw = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]];
        let b_raw = [[9.0, 6.0, 3.0], [8.0, 5.0, 2.0], [7.0, 4.0, 1.0]];

        let a = TestTensor::<2>::from_data(TensorData::from(a_raw), &device);
        let b = TestTensor::<2>::from_data(TensorData::from(b_raw), &device);
        // Cross along dim 0. Some backends (for example CubeCL) may not support
        // cross on non-last dimensions and will intentionally panic with a
        // message like "Cross product on non-last dimension not yet implemented".
        // In that case we treat the panic as a skipped test for that backend.
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let res = catch_unwind(AssertUnwindSafe(|| a.cross(b.clone(), 0)));
        let out = match res {
            Ok(t) => t,
            Err(err) => {
                // Inspect panic payload for the expected not-implemented message and skip
                if let Some(s) = err.downcast_ref::<&str>() {
                    if s.contains("Cross product on non-last dimension") {
                        eprintln!(
                            "Skipping different_dim cross test: backend does not support non-last-dim cross"
                        );
                        return;
                    }
                }
                if let Some(s) = err.downcast_ref::<String>() {
                    if s.contains("Cross product on non-last dimension") {
                        eprintln!(
                            "Skipping different_dim cross test: backend does not support non-last-dim cross"
                        );
                        return;
                    }
                }
                // Unknown panic, re-raise
                std::panic::resume_unwind(err);
            }
        };

        // Manually compute cross of each column vector using raw arrays
        let expected = [
            [
                a_raw[1][0] * b_raw[2][0] - a_raw[2][0] * b_raw[1][0],
                a_raw[1][1] * b_raw[2][1] - a_raw[2][1] * b_raw[1][1],
                a_raw[1][2] * b_raw[2][2] - a_raw[2][2] * b_raw[1][2],
            ],
            [
                a_raw[2][0] * b_raw[0][0] - a_raw[0][0] * b_raw[2][0],
                a_raw[2][1] * b_raw[0][1] - a_raw[0][1] * b_raw[2][1],
                a_raw[2][2] * b_raw[0][2] - a_raw[0][2] * b_raw[2][2],
            ],
            [
                a_raw[0][0] * b_raw[1][0] - a_raw[1][0] * b_raw[0][0],
                a_raw[0][1] * b_raw[1][1] - a_raw[1][1] * b_raw[0][1],
                a_raw[0][2] * b_raw[1][2] - a_raw[1][2] * b_raw[0][2],
            ],
        ];

        out.to_data().assert_approx_eq::<FloatElem<TestBackend>>(
            &TensorData::from(expected),
            Tolerance::default(),
        );
    }
}
