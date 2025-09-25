#[burn_tensor_testgen::testgen(outer)]
mod tests {
    use super::*;
    use burn_tensor::linalg;
    use burn_tensor::{Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_outer_basic() {
        let u = TestTensor::<1>::from([1.0, 2.0, 3.0]);
        let v = TestTensor::<1>::from([4.0, 5.0]);

        let out = linalg::outer(u, v).into_data();
        let expected = TestTensor::<2>::from([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]).into_data();

        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_outer_shapes_only() {
        let device = Default::default();
        let u = TestTensor::<1>::zeros([3], &device);
        let v = TestTensor::<1>::zeros([5], &device);
        let out = linalg::outer(u, v);
        assert_eq!(out.shape().dims(), [3, 5]);
    }

    #[test]
    fn test_outer_asymmetry_and_shapes() {
        let u = TestTensor::<1>::from([1.0, 2.0]);
        let v = TestTensor::<1>::from([3.0, 4.0, 5.0]);

        let uv = linalg::outer(u.clone(), v.clone());
        let vu = linalg::outer(v, u);

        assert_eq!(uv.shape().dims(), [2, 3]);
        assert_eq!(vu.shape().dims(), [3, 2]);
    }

    #[test]
    fn test_outer_zero_left() {
        let device = Default::default();
        let u = TestTensor::<1>::zeros([3], &device);
        let v = TestTensor::<1>::from([7.0, 8.0]);

        let out = linalg::outer(u, v).into_data();
        let expected = TestTensor::<2>::zeros([3, 2], &device).into_data();

        out.assert_eq(&expected, true);
    }

    #[test]
    fn test_outer_zero_right() {
        let device = Default::default();
        let u = TestTensor::<1>::from([1.0, -2.0, 3.0]);
        let v = TestTensor::<1>::zeros([4], &device);

        let out = linalg::outer(u, v).into_data();
        let expected = TestTensor::<2>::zeros([3, 4], &device).into_data();

        out.assert_eq(&expected, true);
    }

    #[test]
    fn test_outer_signs() {
        let u = TestTensor::<1>::from([-1.0, 2.0]);
        let v = TestTensor::<1>::from([3.0, -4.0]);

        let out = linalg::outer(u, v).into_data();
        let expected = TestTensor::<2>::from([[-3.0, 4.0], [6.0, -8.0]]).into_data();

        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_outer_integer_inputs() {
        let u = TestTensorInt::<1>::from([1, 2, 3]);
        let v = TestTensorInt::<1>::from([4, 5]);

        let out = linalg::outer(u, v).into_data();
        let expected = TestTensorInt::<2>::from([[4, 5], [8, 10], [12, 15]]).into_data();

        out.assert_eq(&expected, true);
    }

    #[test]
    fn test_outer_equivalence_to_matmul() {
        let u = TestTensor::<1>::from([1.0, 2.0, 3.0]);
        let v = TestTensor::<1>::from([4.0, 5.0]);

        let out = linalg::outer(u.clone(), v.clone()).into_data();

        let u2 = u.reshape([3, 1]);
        let v2 = v.reshape([1, 2]);
        let out_matmul = u2.matmul(v2).into_data();

        out.assert_approx_eq::<FT>(&out_matmul, Tolerance::default());
    }

    #[test]
    fn test_outer_vector_identity_right_mult() {
        let u = TestTensor::<1>::from([2.0, -1.0]);
        let v = TestTensor::<1>::from([3.0, 4.0]);
        let w = TestTensor::<1>::from([5.0, 6.0]);

        let uv = linalg::outer(u.clone(), v.clone());
        let left = uv.matmul(w.clone().reshape([2, 1])).reshape([2]);

        let v_dot_w = v.dot(w);
        let right = u * v_dot_w;

        left.into_data()
            .assert_approx_eq::<FT>(&right.into_data(), Tolerance::default());
    }

    #[test]
    fn test_outer_length_one_vectors() {
        let u = TestTensor::<1>::from([3.0]);
        let v = TestTensor::<1>::from([4.0, 5.0, 6.0]);

        let out = linalg::outer(u, v).into_data();
        let expected = TestTensor::<2>::from([[12.0, 15.0, 18.0]]).into_data();

        out.assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_outer_large_values() {
        let big = 1.0e10;
        let u = TestTensor::<1>::from([big, -big]);
        let v = TestTensor::<1>::from([big, big]);

        let out = linalg::outer(u, v).into_data();
        let expected =
            TestTensor::<2>::from([[big * big, big * big], [-big * big, -big * big]]).into_data();

        let tol = Tolerance::relative(1e-6).set_half_precision_relative(1e-3);
        out.assert_approx_eq::<FT>(&expected, tol);
    }

    #[test]
    fn test_outer_nan_propagation() {
        let u = TestTensor::<1>::from([f32::NAN, 2.0]);
        let v = TestTensor::<1>::from([3.0, 4.0]);

        let out = linalg::outer(u, v).into_data();

        // as_slice returns Result<&[f32], DataError> on latest Burn
        let values: Vec<f32> = out
            .as_slice::<f32>()
            .expect("outer nan_propagation: as_slice failed")
            .to_vec();

        assert!(values[0].is_nan()); // first row, col0
        assert!(values[1].is_nan()); // first row, col1
        assert_eq!(values[2], 6.0);  // second row, col0
        assert_eq!(values[3], 8.0);  // second row, col1
    }
}