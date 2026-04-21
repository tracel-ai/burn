use super::*;
use burn_tensor::{Distribution, Tolerance, linalg::lu};

// ---------------------------------------------------------------------
// Small Matrices
// ---------------------------------------------------------------------

#[test]
fn test_lu_1x1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[5.0]], &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_2x2() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[4.0, 3.0], [6.0, 3.0]], &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_3x3() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// Special Matrices (using lu)
// ---------------------------------------------------------------------

#[test]
fn test_lu_identity_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_singular_zero_pivot() {
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
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// 2D Tensors (no batch dimension)
// ---------------------------------------------------------------------

#[test]
fn test_lu_2d_square() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 6], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_2d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_2d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_medium_tall() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([256, 128], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_medium_wide() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([128, 256], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// 3D Tensors (1 batch dimension)
// ---------------------------------------------------------------------

#[test]
fn test_lu_3d_square() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 6, 6], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 3, 2>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_3d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 8, 5], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 3, 2>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_3d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 5, 8], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 3, 2>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// 4D Tensors (2 batch dimensions)
// ---------------------------------------------------------------------

#[test]
fn test_lu_4d_square() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 6, 6], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 4, 3>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_4d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 8, 5], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 4, 3>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_4d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 5, 8], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 4, 3>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-2);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// Large Tensors (Triggers Block LU Dispatch)
// ---------------------------------------------------------------------

// The block-dispatch tests below feed an unseeded 500-ish-element random
// matrix through LU + matmul reconstruction. On f16 the per-row error
// grows with n, so a borderline matrix can overshoot a tight absolute
// tolerance (seen: 6.1e-2 diff on a reconstructed value of 0.10 with
// 5e-2 tol). Using 1.5e-1 for f16 on these large sizes keeps the test
// deterministic across seeds while still catching real regressions
// (the f16 LU typo regression in #4738 produced errors of O(1)).

#[test]
fn test_lu_500x500_block_dispatch() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([500, 500], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(1.5e-1);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_500x300_block_dispatch() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([500, 300], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(1.5e-1);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_300x500_block_dispatch() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([300, 500], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 2, 1>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(1.5e-1);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_5x300x300_block_dispatch() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([5, 300, 300], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 3, 2>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(1.5e-1);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_3x300x500_block_dispatch() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 300, 500], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 3, 2>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(1.5e-1);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_lu_3x500x300_block_dispatch() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 500, 300], Distribution::Default, &device);
    let (p, l, u) = lu::<TestBackend, 3, 2>(tensor.clone());
    let plu = p.matmul(l).matmul(u);
    let tolerance = Tolerance::default().set_half_precision_absolute(1.5e-1);
    plu.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// Tensor Check Panics
// ---------------------------------------------------------------------

#[test]
#[should_panic]
fn test_lu_panic_rank_less_than_2() {
    // Fails check: D >= 2
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _ = lu::<TestBackend, 1, 0>(tensor);
}

#[test]
#[should_panic]
fn test_lu_panic_invalid_d1() {
    // Fails check: D - 1 == D1 (2 - 1 != 2)
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let _ = lu::<TestBackend, 2, 2>(tensor);
}
