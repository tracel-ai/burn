use super::*;
use burn_tensor::{Distribution, Tolerance, linalg::qr, s};

const REL: f32 = 5e-3;
const ABS: f32 = 1e-3;

// ---------------------------------------------------------------------
// Small Matrices
// ---------------------------------------------------------------------

#[test]
fn test_qr_1x1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[5.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2x2() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[4.0, 3.0], [6.0, 3.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3x3() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_identity_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_zero_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_maxtrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [0.0, 4.0, 2.0, 6.0],
            [0.0, 2.0, 2.0, 11.0],
            [0.0, 3.0, 9.0, 6.0],
            [0.0, 7.0, 10.0, 9.0],
        ],
        &device,
    );
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_identity_singular_maxtrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([4, 4], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_v0_zero() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[0.0, 4.0, 2.0], [1.0, 2.0, 2.0], [2.0, 3.0, 9.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_diagonilized_col() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[1.0, 4.0, 2.0], [0.0, 2.0, 2.0], [0.0, 3.0, 9.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_5x5_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([5, 5], &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    assert_eq!(q.dims(), [5, 5]);
    assert_eq!(r.dims(), [5, 5]);
}

#[test]
fn test_qr_10x5_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([10, 5], &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    assert_eq!(q.dims(), [10, 5]);
    assert_eq!(r.dims(), [5, 5]);
}

#[test]
fn test_qr_5x10_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([5, 10], &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    assert_eq!(q.dims(), [5, 5]);
    assert_eq!(r.dims(), [5, 10]);
}

#[test]
fn test_qr_2d_square() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 6], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_square_reduced() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 6], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_tall_reduced() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_tall() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([8, 1], &device);
    let mut tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_tall_reduces() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([8, 1], &device);
    let mut tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_wide_reduced() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_wide() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([5, 1], &device);
    let mut tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_wide_reduced() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([5, 1], &device);
    let mut tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), true);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_medium_tall() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([256, 128], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_medium_wide() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([128, 256], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_500x500() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([128, 256], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_500x300() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([128, 256], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_300x500() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([128, 256], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// 3D Tensors (1 batch dimension)
// ---------------------------------------------------------------------

#[test]
fn test_qr_3d_square() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 6, 6], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 8, 5], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 5, 8], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3d_8x6_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 8, 6], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), true);
    assert_eq!(q.dims(), [3, 8, 6]);
    assert_eq!(r.dims(), [3, 6, 6]);
}

// ---------------------------------------------------------------------
// 4D Tensors (2 batch dimensions)
// ---------------------------------------------------------------------

#[test]
fn test_qr_4d_square() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 6, 6], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_square_with_singularities() {
    let device = Default::default();
    let ones = TestTensor::<4>::ones([1, 1, 6, 6], &device);
    let mut tensor = TestTensor::<4>::random([2, 2, 6, 6], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![0..1, 0..1, .., ..], ones.clone());
    tensor = tensor.slice_assign(s![2..3, 1..2, .., ..], ones.clone());
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 8, 5], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_tall_with_singularities() {
    let device = Default::default();
    let ones = TestTensor::<4>::ones([1, 1, 8, 5], &device);
    let mut tensor = TestTensor::<4>::random([2, 2, 8, 5], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![0..1, 0..1, .., ..], ones.clone());
    tensor = tensor.slice_assign(s![2..3, 1..2, .., ..], ones.clone());
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 5, 8], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_wide_with_singularities() {
    let device = Default::default();
    let ones = TestTensor::<4>::ones([1, 1, 5, 8], &device);
    let mut tensor = TestTensor::<4>::random([2, 2, 5, 8], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![0..1, 0..1, .., ..], ones.clone());
    tensor = tensor.slice_assign(s![2..3, 1..2, .., ..], ones.clone());
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_12x4_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 3, 12, 4], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), true);
    assert_eq!(q.dims(), [2, 3, 12, 4]);
    assert_eq!(r.dims(), [2, 3, 4, 4]);
}

// ---------------------------------------------------------------------
// Tensor Check Panics
// ---------------------------------------------------------------------

#[test]
#[should_panic]
fn test_qr_panic_rank_less_than_2() {
    // Fails check: D >= 2
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _ = qr::<1>(tensor, false);
}
