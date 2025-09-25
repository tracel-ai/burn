#[burn_tensor_testgen::testgen(outer_batch)]
mod tests {
    use super::*;
    use burn_tensor::linalg;
    use burn_tensor::{ops::FloatElem, Tolerance};

    type FT = FloatElem<TestBackend>;

    // (1) Basic correctness: two batches
    #[test]
    fn test_outer_batch_basic() {
        // x: (2, 2), y: (2, 3)
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

    // (2) Shape check only
    #[test]
    fn test_outer_batch_shapes() {
        let device = Default::default();
        let x = TestTensor::<2>::zeros([3, 4], &device); // (batch=3, m=4)
        let y = TestTensor::<2>::zeros([3, 5], &device); // (batch=3, n=5)

        let out = linalg::outer_batch(x, y);
        assert_eq!(out.shape().dims(), [3, 4, 5]);
    }

    // (3) Zero cases (left & right)
    #[test]
    fn test_outer_batch_zero_left() {
        let device = Default::default();
        let x = TestTensor::<2>::zeros([2, 3], &device); // (2,3)
        let y = TestTensor::<2>::from([[7.0, 8.0], [9.0, 10.0]]); // (2,2)

        let out = linalg::outer_batch(x, y).into_data();
        let expected = TestTensor::<3>::zeros([2, 3, 2], &device).into_data();

        out.assert_eq(&expected, true);
    }

    #[test]
    fn test_outer_batch_zero_right() {
        let device = Default::default();
        let x = TestTensor::<2>::from([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]); // (2,3)
        let y = TestTensor::<2>::zeros([2, 4], &device); // (2,4)

        let out = linalg::outer_batch(x, y).into_data();
        let expected = TestTensor::<3>::zeros([2, 3, 4], &device).into_data();

        out.assert_eq(&expected, true);
    }

    // (4) Signs
    #[test]
    fn test_outer_batch_signs() {
        let x = TestTensor::<2>::from([[-1.0, 2.0], [3.0, -4.0]]);
        let y = TestTensor::<2>::from([[3.0, -4.0], [-5.0, 6.0]]);
        let out = linalg::outer_batch(x, y).into_data();

        let expected = TestTensor::<3>::from([
            [
                [-3.0, 4.0],
                [6.0, -8.0],
            ],
            [
                [-15.0, 18.0],
                [20.0, -24.0],
            ],
        ])
        .into_data();

        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    // (5) Equivalence with per-sample outer
    #[test]
    fn test_outer_batch_equivalence_to_per_sample_outer() {
        // batch=2, m=2, n=3
        let x = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let y = TestTensor::<2>::from([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]);

        let out_batched = linalg::outer_batch(x.clone(), y.clone());

        for b in 0..2 {
            // select expects a Tensor<_,1,Int> index, not a raw usize
            let idx = TestTensorInt::<1>::from([b as i32]);
            let xb = x.clone().select(0, idx.clone()); // (m,)
            let yb = y.clone().select(0, idx);         // (n,)

            let per = linalg::outer(xb, yb).into_data();

            let idx_b = TestTensorInt::<1>::from([b as i32]);
            let bat = out_batched.clone().select(0, idx_b).into_data();

            bat.assert_approx_eq::<FT>(&per, Tolerance::default());
        }
    }

    // (6) NaN propagation within batches
    #[test]
    fn test_outer_batch_nan_propagation() {
        let x = TestTensor::<2>::from([
            [f32::NAN, 2.0], // batch 0
            [3.0, 4.0],      // batch 1
        ]);
        let y = TestTensor::<2>::from([
            [5.0, 6.0], // batch 0
            [7.0, 8.0], // batch 1
        ]);
        let out = linalg::outer_batch(x, y).into_data();

        // Use slice directly (no Vec) for no_std friendly tests
        let s: &[f32] = out
            .as_slice::<f32>()
            .expect("outer_batch nan_propagation: as_slice failed");

        // batch 0 rows: [NaN, NaN], [10, 12]
        assert!(s[0].is_nan() && s[1].is_nan());
        assert_eq!(s[2], 10.0);
        assert_eq!(s[3], 12.0);

        // batch 1 rows: [21, 24], [28, 32]
        assert_eq!(s[4], 21.0);
        assert_eq!(s[5], 24.0);
        assert_eq!(s[6], 28.0);
        assert_eq!(s[7], 32.0);
    }
}
