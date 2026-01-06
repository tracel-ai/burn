use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::linalg;

#[test]
fn test_max_min_abs() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

    let expected = TestTensor::<2>::from([[3., 4.]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::LInf, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::max_abs_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);

    let expected = TestTensor::<2>::from([[1., 2.]]).into_data();
    linalg::vector_norm(x.clone(), -f64::INFINITY, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::vector_norm(x.clone(), f64::NEG_INFINITY, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::min_abs_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);

    let expected = TestTensor::<2>::from([[2.], [4.]]).into_data();
    linalg::vector_norm(x.clone(), f64::INFINITY, 1)
        .into_data()
        .assert_eq(&expected, true);
    linalg::max_abs_norm(x.clone(), 1)
        .into_data()
        .assert_eq(&expected, true);

    let expected = TestTensor::<2>::from([[1.], [3.]]).into_data();
    linalg::vector_norm(x.clone(), -f64::INFINITY, 1)
        .into_data()
        .assert_eq(&expected, true);
    linalg::vector_norm(x.clone(), f64::NEG_INFINITY, 1)
        .into_data()
        .assert_eq(&expected, true);
    linalg::min_abs_norm(x, 1)
        .into_data()
        .assert_eq(&expected, true);

    // Test with integer tensor
    let z = TestTensorInt::<2>::from([[1, 2], [3, 4]]);

    linalg::max_abs_norm(z.clone(), 0)
        .into_data()
        .assert_eq(&TestTensorInt::<2>::from([[3, 4]]).into_data(), true);
    linalg::max_abs_norm(z.clone(), 1)
        .into_data()
        .assert_eq(&TestTensorInt::<2>::from([[2], [4]]).into_data(), true);

    linalg::min_abs_norm(z.clone(), 0)
        .into_data()
        .assert_eq(&TestTensorInt::<2>::from([[1, 2]]).into_data(), true);
    linalg::min_abs_norm(z, 1)
        .into_data()
        .assert_eq(&TestTensorInt::<2>::from([[1], [3]]).into_data(), true);
}

#[test]
fn test_l0_norm() {
    let x = TestTensor::<2>::from([[1.0, -2.0, 0.], [0.0, 0., 4.]]);

    let expected = TestTensor::<2>::from([[1., 1., 1.]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L0, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::l0_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);

    let expected = TestTensor::<2>::from([[2.], [1.]]).into_data();
    linalg::vector_norm(x.clone(), 0.0, 1)
        .into_data()
        .assert_eq(&expected, true);
    linalg::l0_norm(x.clone(), 1)
        .into_data()
        .assert_eq(&expected, true);

    // Test with integer tensor
    let z = TestTensorInt::<2>::from([[1, -2, 0], [0, 0, 4]]);

    linalg::l0_norm(z.clone(), 0)
        .into_data()
        .assert_eq(&TestTensor::<2>::from([[1, 1, 1]]).int().into_data(), true);
    linalg::l0_norm(z.clone(), 1)
        .into_data()
        .assert_eq(&TestTensor::<2>::from([[2], [1]]).int().into_data(), true);
}

#[test]
fn test_l1_norm() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

    let expected = TestTensor::<2>::from([[4.0, 6.0]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L1, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::l1_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);

    let expected = TestTensor::<2>::from([[3.0], [7.0]]).into_data();
    linalg::vector_norm(x.clone(), 1.0, 1)
        .into_data()
        .assert_eq(&expected, true);
    linalg::l1_norm(x.clone(), 1)
        .into_data()
        .assert_eq(&expected, true);
}

#[test]
fn test_lp_norm() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(2e-3);

    let expected = TestTensor::<2>::from([[3.0365891, 4.1601677]]).into_data();
    linalg::vector_norm(x.clone(), 3, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::lp_norm(x.clone(), 3., 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_l2_norm() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(1e-3);

    let expected = TestTensor::<2>::from([[3.16227766, 4.47213595]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L2, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::l2_norm(x.clone(), 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TestTensor::<2>::from([[2.23606798], [5.0]]).into_data();
    linalg::vector_norm(x.clone(), 2.0, 1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::l2_norm(x.clone(), 1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_normalize() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

    let expected = TensorData::from([[1. / 4., 2. / 6.], [3. / 4., 4. / 6.]]);
    let output = linalg::vector_normalize(x.clone(), 1.0, 0, 0.25).into_data();
    output.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[1. / 5., 2. / 6.], [3. / 5., 4. / 6.]]);
    let output = linalg::vector_normalize(x.clone(), 1.0, 0, 5.0).into_data();
    output.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
