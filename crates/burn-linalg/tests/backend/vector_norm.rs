use super::*;
use burn_core::tensor::Shape;
use burn_core::tensor::TensorData;
use burn_core::tensor::Tolerance;
use burn_linalg as linalg;

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
    let x = TestTensor::<2>::from([[1., -2., 0.], [0., 3., 4.]]);
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(2e-3);

    fn lp_norm_naive<const D: usize>(x: Tensor<D>, p: f64, dim: usize) -> Tensor<D> {
        x.abs().powf_scalar(p).sum_dim(dim).powf_scalar(1. / p)
    }

    // Arbitrary P
    let expected = TestTensor::<2>::from([[1.0, 3.2710664, 4.0]]).into_data();
    linalg::vector_norm(x.clone(), 3, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::lp_norm(x.clone(), 3., 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    // L0
    let expected = TestTensor::<2>::from([[1., 2., 1.]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L0, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::l0_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::lp_norm(x.clone(), 0.0, 0)
        .into_data()
        .assert_eq(&expected, true);

    // L1
    let expected = TestTensor::<2>::from([[1.0, 5.0, 4.0]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L1, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::l1_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);
    lp_norm_naive(x.clone(), 1.0, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::lp_norm(x.clone(), 1.0, 0)
        .into_data()
        .assert_eq(&expected, true);

    // L2
    let expected = TestTensor::<2>::from([[1.0, 3.6055512, 4.0]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L2, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::l2_norm(x.clone(), 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    lp_norm_naive(x.clone(), 2.0, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::lp_norm(x.clone(), 2.0, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    // LInf
    let expected = TestTensor::<2>::from([[1.0, 3.0, 4.0]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::LInf, 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::max_abs_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::lp_norm(x.clone(), f64::INFINITY, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    // LNegInf
    let expected = TestTensor::<2>::from([[0.0, 2.0, 0.0]]).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::LNegInf, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
    linalg::min_abs_norm(x.clone(), 0)
        .into_data()
        .assert_eq(&expected, true);
    linalg::lp_norm(x.clone(), f64::NEG_INFINITY, 0)
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

#[test]
fn test_negative_dimension() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);
    let tolerance = Tolerance::default();

    let expected = linalg::vector_norm(x.clone(), linalg::Norm::L2, 1).into_data();
    linalg::vector_norm(x.clone(), linalg::Norm::L2, -1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = linalg::lp_norm(x.clone(), 3.0, 1).into_data();
    linalg::lp_norm(x.clone(), 3.0, -1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = linalg::vector_normalize(x.clone(), 1.0, 1, 1e-8).into_data();
    linalg::vector_normalize(x.clone(), 1.0, -1, 1e-8)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = linalg::l0_norm(x.clone(), 1).into_data();
    linalg::l0_norm(x.clone(), -1)
        .into_data()
        .assert_eq(&expected, true);

    let expected = linalg::l1_norm(x.clone(), 1).into_data();
    linalg::l1_norm(x.clone(), -1)
        .into_data()
        .assert_eq(&expected, true);

    let expected = linalg::l2_norm(x.clone(), 1).into_data();
    linalg::l2_norm(x.clone(), -1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = linalg::max_abs_norm(x.clone(), 1).into_data();
    linalg::max_abs_norm(x.clone(), -1)
        .into_data()
        .assert_eq(&expected, true);

    let expected = linalg::min_abs_norm(x.clone(), 1).into_data();
    linalg::min_abs_norm(x, -1)
        .into_data()
        .assert_eq(&expected, true);
}

#[test]
fn test_spatial_multi_axis_reduction() {
    let device = Default::default();
    let x = TestTensor::<4>::ones([2, 3, 4, 5], &device);

    let out_l1 = linalg::l1_norm_dims(x.clone(), &[2, 3]);
    assert_eq!(out_l1.shape(), Shape::new([2, 3, 1, 1]));

    let out_l2 = linalg::l2_norm_dims(x.clone(), &[2, 3]);
    assert_eq!(out_l2.shape(), Shape::new([2, 3, 1, 1]));

    let out_lp = linalg::lp_norm_dims(x.clone(), 3.0, &[2, 3]);
    assert_eq!(out_lp.shape(), Shape::new([2, 3, 1, 1]));

    let out_max = linalg::max_abs_norm_dims(x.clone(), &[2, 3]);
    assert_eq!(out_max.shape(), Shape::new([2, 3, 1, 1]));

    let out_min = linalg::min_abs_norm_dims(x.clone(), &[2, 3]);
    assert_eq!(out_min.shape(), Shape::new([2, 3, 1, 1]));

    let out_l0 = linalg::l0_norm_dims(x.clone(), &[2, 3]);
    assert_eq!(out_l0.shape(), Shape::new([2, 3, 1, 1]));

    let out_vec = linalg::vector_norm_dims(x, linalg::Norm::L2, &[2, 3]);
    assert_eq!(out_vec.shape(), Shape::new([2, 3, 1, 1]));
}

#[test]
fn test_multi_axis_numerical_equivalence() {
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(2e-3);
    let x = TestTensor::<3>::from([
        [
            [1.0, -2.0, 3.0, -4.0],
            [5.0, -6.0, 7.0, -8.0],
            [9.0, -10.0, 11.0, -12.0],
        ],
        [
            [-13.0, 14.0, -15.0, 16.0],
            [-17.0, 18.0, -19.0, 20.0],
            [-21.0, 22.0, -23.0, 24.0],
        ],
    ]);

    // L1 norm
    let expected_l1 = x.clone().abs().sum_dims(&[1, 2]).into_data();
    linalg::l1_norm_dims(x.clone(), &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_l1, tolerance);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::L1, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_l1, tolerance);

    // L2 norm
    let expected_l2 = x.clone().square().sum_dims(&[1, 2]).sqrt().into_data();
    linalg::l2_norm_dims(x.clone(), &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_l2, tolerance);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::L2, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_l2, tolerance);

    // Lp norm: even integer p = 4.0
    let expected_p4 = x
        .clone()
        .powi_scalar(4)
        .sum_dims(&[1, 2])
        .powf_scalar(1.0 / 4.0)
        .into_data();
    linalg::lp_norm_dims(x.clone(), 4.0, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_p4, tolerance);
    linalg::vector_norm_dims(x.clone(), 4, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_p4, tolerance);

    // Lp norm: odd integer p = 3.0
    let expected_p3 = x
        .clone()
        .abs()
        .powf_scalar(3.0)
        .sum_dims(&[1, 2])
        .powf_scalar(1.0 / 3.0)
        .into_data();
    linalg::lp_norm_dims(x.clone(), 3.0, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_p3, tolerance);
    linalg::vector_norm_dims(x.clone(), 3, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_p3, tolerance);

    // Lp norm: non-integer p = 1.5
    let expected_p1_5 = x
        .clone()
        .abs()
        .powf_scalar(1.5)
        .sum_dims(&[1, 2])
        .powf_scalar(1.0 / 1.5)
        .into_data();
    linalg::lp_norm_dims(x.clone(), 1.5, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_p1_5, tolerance);
    linalg::vector_norm_dims(x.clone(), 1.5, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_p1_5, tolerance);

    // MaxAbs (LInf) norm
    let expected_max = x.clone().abs().max_dim(1).max_dim(2).into_data();
    linalg::max_abs_norm_dims(x.clone(), &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_max, tolerance);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::LInf, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_max, tolerance);

    // MinAbs (LNegInf) norm
    let expected_min = x.clone().abs().min_dim(1).min_dim(2).into_data();
    linalg::min_abs_norm_dims(x.clone(), &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_min, tolerance);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::LNegInf, &[1, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_min, tolerance);

    // L0 norm
    let expected_l0 = x
        .clone()
        .zeros_like()
        .mask_fill(x.clone().not_equal_scalar(0), 1)
        .sum_dims(&[1, 2])
        .into_data();
    linalg::l0_norm_dims(x.clone(), &[1, 2])
        .into_data()
        .assert_eq(&expected_l0, true);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::L0, &[1, 2])
        .into_data()
        .assert_eq(&expected_l0, true);
}

#[test]
fn test_single_vs_multi_equivalence() {
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(1e-3);
    let x = TestTensor::<2>::from([[1., -2.], [3., 4.]]);

    // L1
    let single = linalg::l1_norm(x.clone(), 1).into_data();
    let multi = linalg::l1_norm_dims(x.clone(), &[1]).into_data();
    single.assert_eq(&multi, true);

    // L2
    let single = linalg::l2_norm(x.clone(), 1).into_data();
    let multi = linalg::l2_norm_dims(x.clone(), &[1]).into_data();
    single.assert_approx_eq::<FloatElem>(&multi, tolerance);

    // Lp
    let single = linalg::lp_norm(x.clone(), 3.0, 1).into_data();
    let multi = linalg::lp_norm_dims(x.clone(), 3.0, &[1]).into_data();
    single.assert_approx_eq::<FloatElem>(&multi, tolerance);

    // MaxAbs
    let single = linalg::max_abs_norm(x.clone(), 1).into_data();
    let multi = linalg::max_abs_norm_dims(x.clone(), &[1]).into_data();
    single.assert_eq(&multi, true);

    // MinAbs
    let single = linalg::min_abs_norm(x.clone(), 1).into_data();
    let multi = linalg::min_abs_norm_dims(x.clone(), &[1]).into_data();
    single.assert_eq(&multi, true);

    // L0
    let single = linalg::l0_norm(x.clone(), 1).into_data();
    let multi = linalg::l0_norm_dims(x.clone(), &[1]).into_data();
    single.assert_eq(&multi, true);

    // Vector norm
    let single = linalg::vector_norm(x.clone(), linalg::Norm::L2, 1).into_data();
    let multi = linalg::vector_norm_dims(x, linalg::Norm::L2, &[1]).into_data();
    single.assert_approx_eq::<FloatElem>(&multi, tolerance);
}

#[test]
fn test_multi_axis_negative_dimensions() {
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(1e-3);
    let x = TestTensor::<3>::from([
        [
            [1.0, -2.0, 3.0, -4.0],
            [5.0, -6.0, 7.0, -8.0],
            [9.0, -10.0, 11.0, -12.0],
        ],
        [
            [-13.0, 14.0, -15.0, 16.0],
            [-17.0, 18.0, -19.0, 20.0],
            [-21.0, 22.0, -23.0, 24.0],
        ],
    ]);

    let expected = linalg::l2_norm_dims(x.clone(), &[1, 2]).into_data();
    linalg::l2_norm_dims(x.clone(), &[-2, -1])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = linalg::l1_norm_dims(x.clone(), &[1, 2]).into_data();
    linalg::l1_norm_dims(x.clone(), &[-2, -1])
        .into_data()
        .assert_eq(&expected, true);

    let expected = linalg::max_abs_norm_dims(x.clone(), &[1, 2]).into_data();
    linalg::max_abs_norm_dims(x.clone(), &[-2, -1])
        .into_data()
        .assert_eq(&expected, true);

    let expected = linalg::lp_norm_dims(x.clone(), 3.0, &[1, 2]).into_data();
    linalg::lp_norm_dims(x, 3.0, &[-2, -1])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_empty_dims() {
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(1e-3);
    let x = TestTensor::<1>::from([-2.0, 0.0, 3.0]);
    let expected_abs = TestTensor::<1>::from([2.0, 0.0, 3.0]).into_data();
    let expected_l0 = TestTensor::<1>::from([1.0, 0.0, 1.0]).into_data();

    // L1 norm: elementwise absolute value
    linalg::l1_norm_dims(x.clone(), &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs, true);

    // L2 norm: elementwise absolute value
    linalg::l2_norm_dims(x.clone(), &[] as &[usize])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_abs, tolerance);

    // L_infinity norm: elementwise absolute value
    linalg::max_abs_norm_dims(x.clone(), &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs, true);

    // L_neg_infinity norm: elementwise absolute value
    linalg::min_abs_norm_dims(x.clone(), &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs, true);

    // Lp norm (odd and even p): elementwise absolute value
    linalg::lp_norm_dims(x.clone(), 3.0, &[] as &[usize])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_abs, tolerance);
    linalg::lp_norm_dims(x.clone(), 4.0, &[] as &[usize])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_abs, tolerance);

    // Vector norm dispatch
    linalg::vector_norm_dims(x.clone(), linalg::Norm::L1, &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs, true);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::L2, &[] as &[usize])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_abs, tolerance);
    linalg::vector_norm_dims(x.clone(), linalg::Norm::LInf, &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs, true);

    // L0 norm: elementwise non-zero indicator mask
    linalg::l0_norm_dims(x.clone(), &[] as &[usize])
        .into_data()
        .assert_eq(&expected_l0, true);
    linalg::vector_norm_dims(x, linalg::Norm::L0, &[] as &[usize])
        .into_data()
        .assert_eq(&expected_l0, true);

    // Multi-dimensional tensor with empty dims
    let x_2d = TestTensor::<2>::from([[-2.0, 0.0], [3.0, -4.0]]);
    let expected_abs_2d = TestTensor::<2>::from([[2.0, 0.0], [3.0, 4.0]]).into_data();
    let expected_l0_2d = TestTensor::<2>::from([[1.0, 0.0], [1.0, 1.0]]).into_data();

    linalg::l1_norm_dims(x_2d.clone(), &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs_2d, true);
    linalg::l2_norm_dims(x_2d.clone(), &[] as &[usize])
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_abs_2d, tolerance);
    linalg::max_abs_norm_dims(x_2d.clone(), &[] as &[usize])
        .into_data()
        .assert_eq(&expected_abs_2d, true);
    linalg::l0_norm_dims(x_2d, &[] as &[usize])
        .into_data()
        .assert_eq(&expected_l0_2d, true);
}

#[test]
#[should_panic(expected = "Vector Normalize")]
fn test_vector_normalize_panic() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);
    let _ = linalg::vector_normalize(x, 1.0, 5, 1e-5);
}

#[test]
#[should_panic(expected = "Vector Norm")]
fn test_vector_norm_panic() {
    let x = TestTensor::<2>::from([[1., 2.], [3., 4.]]);
    let _ = linalg::vector_norm(x, linalg::Norm::L2, 5);
}
