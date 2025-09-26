#[burn_tensor_testgen::testgen(outer_batch)]
mod tests {
    use super::*;
    use burn_tensor::linalg;
    use burn_tensor::{Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;

    // (1) Basic correctness: two batches
    #[test]
    fn test_outer_batch_basic() {
        let x = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let y = TestTensor::<2>::from([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]);
        let out = linalg::outer_batch(x, y).into_data();

        let expected = TestTensor::<3>::from([
            [[5.0, 6.0, 7.0], [10.0, 12.0, 14.0]],
            [[24.0, 27.0, 30.0], [32.0, 36.0, 40.0]],
        ])
        .into_data();

        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    // (2) Shape check
    #[test]
    fn test_outer_batch_shapes() {
        let device = Default::default();
        let x = TestTensor::<2>::zeros([3, 4], &device);
        let y = TestTensor::<2>::zeros([3, 5], &device);
        let out = linalg::outer_batch(x, y);
        assert_eq!(out.shape().dims(), [3, 4, 5]);
    }

    // (3) Zero cases
    #[test]
    fn test_outer_batch_zero_left() {
        let device = Default::default();
        let x = TestTensor::<2>::zeros([2, 3], &device);
        let y = TestTensor::<2>::from([[7.0, 8.0], [9.0, 10.0]]);

        let out = linalg::outer_batch(x, y).into_data();
        let expected = TestTensor::<3>::zeros([2, 3, 2], &device).into_data();

        // tolerate -0.0 vs 0.0
        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_outer_batch_zero_right() {
        let device = Default::default();
        let x = TestTensor::<2>::from([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]);
        let y = TestTensor::<2>::zeros([2, 4], &device);

        let out = linalg::outer_batch(x, y).into_data();
        let expected = TestTensor::<3>::zeros([2, 3, 4], &device).into_data();

        // tolerate -0.0 vs 0.0
        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    // (4) Signs
    #[test]
    fn test_outer_batch_signs() {
        let x = TestTensor::<2>::from([[-1.0, 2.0], [3.0, -4.0]]);
        let y = TestTensor::<2>::from([[3.0, -4.0], [-5.0, 6.0]]);
        let out = linalg::outer_batch(x, y).into_data();

        let expected =
            TestTensor::<3>::from([[[-3.0, 4.0], [6.0, -8.0]], [[-15.0, 18.0], [20.0, -24.0]]])
                .into_data();

        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    // (5) Equivalence with per-sample outer
    #[test]
    fn test_outer_batch_equivalence_to_per_sample_outer() {
        let x = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let y = TestTensor::<2>::from([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]);

        let batched = linalg::outer_batch(x.clone(), y.clone());

        for b in 0..2 {
            let idx = TestTensorInt::<1>::from([b as i32]);

            let xb2d = x.clone().select(0, idx.clone());
            let yb2d = y.clone().select(0, idx);

            let dims_x: [usize; 2] = xb2d.shape().dims();
            let dims_y: [usize; 2] = yb2d.shape().dims();
            let (m, n) = (dims_x[1], dims_y[1]);

            let per = linalg::outer(xb2d.reshape([m]), yb2d.reshape([n]));

            let bat3d = batched
                .clone()
                .select(0, TestTensorInt::<1>::from([b as i32]));

            let per_len = per.shape().num_elements();
            let per_flat = per.reshape([per_len]).into_data();

            let bat_len = bat3d.shape().num_elements();
            let bat_flat = bat3d.reshape([bat_len]).into_data();

            bat_flat.assert_approx_eq::<FT>(&per_flat, Tolerance::default());
        }
    }

    // (6) NaN propagation
    #[test]
    fn test_outer_batch_nan_propagation() {
        let x = TestTensor::<2>::from([[f32::NAN, 2.0], [3.0, 4.0]]);
        let y = TestTensor::<2>::from([[5.0, 6.0], [7.0, 8.0]]);
        let out = linalg::outer_batch(x, y).into_data();

        let s: &[f32] = out.as_slice::<f32>().expect("outer_batch as_slice failed");

        assert!(s[0].is_nan() && s[1].is_nan());
        assert_eq!(s[2], 10.0);
        assert_eq!(s[3], 12.0);

        assert_eq!(s[4], 21.0);
        assert_eq!(s[5], 24.0);
        assert_eq!(s[6], 28.0);
        assert_eq!(s[7], 32.0);
    }

    // (7) Mismatched batch size should panic
    #[test]
    #[should_panic]
    fn test_outer_batch_mismatched_batches_panics() {
        let device = Default::default();
        let x = TestTensor::<2>::zeros([2, 3], &device);
        let y = TestTensor::<2>::zeros([3, 4], &device);
        let _ = linalg::outer_batch(x, y);
    }

    // (8) Duplicate safeguard
    #[test]
    #[should_panic]
    fn test_outer_batch_mismatched_batch_panics() {
        let device = Default::default();
        let x = TestTensor::<2>::zeros([2, 3], &device);
        let y = TestTensor::<2>::zeros([3, 4], &device);
        let _ = linalg::outer_batch(x, y);
    }
}
