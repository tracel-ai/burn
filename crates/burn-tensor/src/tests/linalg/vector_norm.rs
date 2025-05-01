#[burn_tensor_testgen::testgen(vector_norm)]
mod tests {
    use super::*;
    use burn_tensor::Tolerance;
    use burn_tensor::backend::Backend;
    use burn_tensor::linalg;

    #[test]
    fn test_pos_inf() {
        let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

        linalg::vector_norm(x.clone(), f64::INFINITY, 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[3., 4.]]).into_data(), true);
        linalg::linf_norm(x.clone(), 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[3., 4.]]).into_data(), true);

        linalg::vector_norm(x.clone(), -f64::INFINITY, 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1., 2.]]).into_data(), true);
        linalg::vector_norm(x.clone(), f64::NEG_INFINITY, 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1., 2.]]).into_data(), true);
        linalg::lneg_inf_norm(x.clone(), 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1., 2.]]).into_data(), true);

        linalg::vector_norm(x.clone(), f64::INFINITY, 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[2.], [4.]]).into_data(), true);
        linalg::linf_norm(x.clone(), 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[2.], [4.]]).into_data(), true);

        linalg::vector_norm(x.clone(), -f64::INFINITY, 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1.], [3.]]).into_data(), true);
        linalg::vector_norm(x.clone(), f64::NEG_INFINITY, 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1.], [3.]]).into_data(), true);
        linalg::lneg_inf_norm(x.clone(), 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1.], [3.]]).into_data(), true);
    }

    #[test]
    fn test_zero() {
        let x = TestTensor::<2>::from([[1.0, -2.0, 0.], [0.0, 0., 4.]]);

        linalg::vector_norm(x.clone(), 0.0, 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1., 1., 1.]]).into_data(), true);
        linalg::l0_norm(x.clone(), 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[1., 1., 1.]]).into_data(), true);

        linalg::vector_norm(x.clone(), 0.0, 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[2.], [1.]]).into_data(), true);
        linalg::l0_norm(x.clone(), 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[2.], [1.]]).into_data(), true);
    }

    #[test]
    fn test_l1_norm() {
        let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

        linalg::vector_norm(x.clone(), 1.0, 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[4.0, 6.0]]).into_data(), true);
        linalg::l1_norm(x.clone(), 0)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[4.0, 6.0]]).into_data(), true);

        linalg::vector_norm(x.clone(), 1.0, 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[3.0], [7.0]]).into_data(), true);
        linalg::l1_norm(x.clone(), 1)
            .into_data()
            .assert_eq(&TestTensor::<2>::from([[3.0], [7.0]]).into_data(), true);
    }

    #[test]
    fn test_l2_norm() {
        let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);
        let tolerance = Tolerance::<f32>::absolute(1e-5);

        linalg::vector_norm(x.clone(), 2.0, 0)
            .into_data()
            .assert_approx_eq(
                &TestTensor::<2>::from([[3.1622776601683795, 4.47213595499958]]).into_data(),
                tolerance,
            );
        linalg::l2_norm(x.clone(), 0).into_data().assert_approx_eq(
            &TestTensor::<2>::from([[3.1622776601683795, 4.47213595499958]]).into_data(),
            tolerance,
        );

        linalg::vector_norm(x.clone(), 2.0, 1)
            .into_data()
            .assert_approx_eq(
                &TestTensor::<2>::from([[2.23606797749979], [5.0]]).into_data(),
                tolerance,
            );
        linalg::l2_norm(x.clone(), 1).into_data().assert_approx_eq(
            &TestTensor::<2>::from([[2.23606797749979], [5.0]]).into_data(),
            tolerance,
        );
    }
}
