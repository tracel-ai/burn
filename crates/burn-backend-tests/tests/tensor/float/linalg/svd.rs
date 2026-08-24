use super::*;
use burn_tensor::TensorData;
use burn_tensor::{DType, Distribution, Tolerance, linalg::svd};

const REL: f32 = 5e-3;
const ABS: f32 = 1e-3;

fn tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2)
}

/// Singular values read back as plain scalars (ordering and sign checks).
fn singular_values(s: TestTensor<1>) -> Vec<f32> {
    s.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

/// Assert `A = U diag(S) Vt` (the defining property of the decomposition).
fn assert_reconstruction<const D: usize, const D1: usize>(
    a: &TestTensor<D>,
    u: &TestTensor<D>,
    s: &TestTensor<D1>,
    vt: &TestTensor<D>,
) {
    // Scale the columns of U by S ([.., k] becomes [.., 1, k]) and multiply.
    let recon = u
        .clone()
        .mul(s.clone().unsqueeze_dim(D - 2))
        .matmul(vt.clone());
    recon
        .into_data()
        .assert_approx_eq::<FloatElem>(&a.clone().into_data(), tolerance());
}

// ---------------------------------------------------------------------
// Reconstruction and shapes
// ---------------------------------------------------------------------

#[test]
fn test_svd_rectangular_reconstruction() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [3, 2]);
    assert_eq!(s.dims(), [2]);
    assert_eq!(vt.dims(), [2, 2]);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
}

#[test]
fn test_svd_square_reconstruction() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [3, 3]);
    assert_eq!(s.dims(), [3]);
    assert_eq!(vt.dims(), [3, 3]);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
}

#[test]
fn test_svd_wide_reconstruction() {
    // m < n goes through the transposed formulation; U is [m, k], Vt is [k, n].
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 2]);
    assert_eq!(s.dims(), [2]);
    assert_eq!(vt.dims(), [2, 3]);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
}

#[test]
fn test_svd_single_row() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[3.0, 4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [1, 1]);
    assert_eq!(s.dims(), [1]);
    assert_eq!(vt.dims(), [1, 2]);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
    let expected = TestTensor::<1>::from_data([5.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_single_column() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[3.0], [4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 1]);
    assert_eq!(s.dims(), [1]);
    assert_eq!(vt.dims(), [1, 1]);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
    let expected = TestTensor::<1>::from_data([5.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_1x1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[-7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
    let expected = TestTensor::<1>::from_data([7.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_negative_determinant_2x2() {
    // Regression: a handedness fix once negated one Vt row without the
    // matching U column, breaking every det < 0 input.
    let device = Default::default();
    for a in [
        [[1.0, 2.0], [3.0, 4.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 2.0], [2.0, 1.0]],
    ] {
        let tensor = TestTensor::<3>::from_data([a], &device);
        let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
        assert_reconstruction::<3, 2>(&tensor, &u, &s, &vt);
    }
}

#[test]
fn test_svd_batched_reconstruction_and_shapes() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        &device,
    );
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 3, 3]);
    assert_eq!(s.dims(), [2, 3]);
    assert_eq!(vt.dims(), [2, 3, 3]);
    assert_reconstruction::<3, 2>(&tensor, &u, &s, &vt);
}

#[test]
fn test_svd_batched_mixed_values() {
    // Batch 0 is rank-deficient: its null-space factor is arbitrary, so only
    // the singular values are compared.
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [[[1.0, 2.0], [2.0, 4.0]], [[1.0, 0.0], [0.0, 3.0]]],
        &device,
    );
    let (_u, s, _vt) = svd::<3, 2>(tensor, 15);
    // Batch 0: rank-1 [[1, 2], [2, 4]] gives sigma = [5, 0].
    let expected = TestTensor::<2>::from_data([[5.0, 0.0], [3.0, 1.0]], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_batched_wide() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        ],
        &device,
    );
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 2, 2]);
    assert_eq!(s.dims(), [2, 2]);
    assert_eq!(vt.dims(), [2, 2, 3]);
    assert_reconstruction::<3, 2>(&tensor, &u, &s, &vt);
}

#[test]
fn test_svd_large_batch_threaded_path() {
    // Big enough for the batch-parallel host path (batch * m * n >= 4096).
    let device = Default::default();
    let tensor = TestTensor::<3>::random([4, 48, 32], Distribution::Normal(0.0, 1.0), &device);
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_eq!(s.dims(), [4, 32]);
    assert_reconstruction::<3, 2>(&tensor, &u, &s, &vt);
}

// ---------------------------------------------------------------------
// Orthonormality of the factors
// ---------------------------------------------------------------------

#[test]
fn test_svd_orthonormal_square() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (u, _s, vt) = svd::<2, 1>(tensor, 15);
    let utu = u.clone().transpose().matmul(u);
    utu.into_data()
        .assert_approx_eq::<FloatElem>(&TestTensor::<2>::eye(3, &device).into_data(), tolerance());
    let vt_t = vt.clone().transpose();
    let vvt = vt.matmul(vt_t);
    vvt.into_data()
        .assert_approx_eq::<FloatElem>(&TestTensor::<2>::eye(3, &device).into_data(), tolerance());
}

#[test]
fn test_svd_orthonormal_rectangular() {
    // U [m, k] has orthonormal columns, Vt [k, n] has orthonormal rows.
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 4], Distribution::Normal(0.0, 1.0), &device);
    let (u, _s, vt) = svd::<2, 1>(tensor, 15);
    let utu = u.clone().transpose().matmul(u);
    utu.into_data()
        .assert_approx_eq::<FloatElem>(&TestTensor::<2>::eye(4, &device).into_data(), tolerance());
    let vt_t = vt.clone().transpose();
    let vvt = vt.matmul(vt_t);
    vvt.into_data()
        .assert_approx_eq::<FloatElem>(&TestTensor::<2>::eye(4, &device).into_data(), tolerance());
}

#[test]
fn test_svd_zero_matrix_factors_orthonormal() {
    // Even with every singular value at zero the returned factors are
    // orthonormal and contain no NaNs.
    let device = Default::default();
    let tensor = TestTensor::<2>::zeros([4, 4], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
    let utu = u.clone().transpose().matmul(u);
    utu.into_data()
        .assert_approx_eq::<FloatElem>(&TestTensor::<2>::eye(4, &device).into_data(), tolerance());
    let vt_t = vt.clone().transpose();
    let vvt = vt.matmul(vt_t);
    vvt.into_data()
        .assert_approx_eq::<FloatElem>(&TestTensor::<2>::eye(4, &device).into_data(), tolerance());
    let zeros = TestTensor::<1>::from_data([0.0, 0.0, 0.0, 0.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&zeros.into_data(), tolerance());
}

// ---------------------------------------------------------------------
// Singular values: known cases, ordering, invariances
// ---------------------------------------------------------------------

#[test]
fn test_svd_diagonal_values() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[3.0, 0.0], [0.0, 1.0]], &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let expected = TestTensor::<1>::from_data([3.0, 1.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_identity_values() {
    let device = Default::default();
    let tensor = TestTensor::<2>::eye(3, &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let expected = TestTensor::<1>::from_data([1.0, 1.0, 1.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_rank_one_values() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let expected = TestTensor::<1>::from_data([5.0, 0.0], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_clustered_values() {
    // Near-equal singular values are the stress case for shifted QR.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[2.0, 0.05], [0.05, 1.95]], &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    // Eigenvalues of the symmetric input A (PSD): (3.95 +- sqrt(0.0125)) / 2.
    let expected = TestTensor::<1>::from_data([2.030_901_7, 1.919_098_3], &device);
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_hilbert_ill_conditioned_values() {
    // Hilbert 5x5 has a condition number around 5e5; the values below are
    // the LAPACK reference singular values (numpy, f64).
    let device = Default::default();
    let flat: Vec<f32> = (0..25)
        .map(|idx| 1.0 / (idx / 5 + idx % 5 + 1) as f32)
        .collect();
    let tensor = TestTensor::<2>::from_data(TensorData::new(flat, [5, 5]), &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let expected = TestTensor::<1>::from_data(
        [
            1.567_050_7,
            0.208_534_22,
            0.011_407_492,
            0.000_305_898_04,
            0.000_003_287_928_8,
        ],
        &device,
    );
    s.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance());
}

#[test]
fn test_svd_values_descending_non_negative() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([7, 5], Distribution::Normal(0.0, 1.0), &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let vals = singular_values(s);
    assert_eq!(vals.len(), 5);
    for pair in vals.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "singular values must be sorted descending: {vals:?}"
        );
    }
    assert!(
        vals.iter().all(|&v| v >= 0.0),
        "singular values must be non-negative: {vals:?}"
    );
}

#[test]
fn test_svd_scaling_invariance() {
    // sigma(2A) = 2 sigma(A).
    let device = Default::default();
    let a = TestTensor::<2>::random([5, 4], Distribution::Normal(0.0, 1.0), &device);
    let (_u1, s1, _vt1) = svd::<2, 1>(a.clone(), 15);
    let (_u2, s2, _vt2) = svd::<2, 1>(a.mul_scalar(2.0), 15);
    let doubled = s1.mul_scalar(2.0);
    s2.into_data()
        .assert_approx_eq::<FloatElem>(&doubled.into_data(), tolerance());
}

#[test]
fn test_svd_transpose_invariance() {
    // sigma(A) = sigma(A^T); the transpose case also drives the wide-input
    // path of the implementation.
    let device = Default::default();
    let a = TestTensor::<2>::random([4, 6], Distribution::Normal(0.0, 1.0), &device);
    let (_u1, s1, _vt1) = svd::<2, 1>(a.clone(), 15);
    let (_u2, s2, _vt2) = svd::<2, 1>(a.transpose(), 15);
    s2.into_data()
        .assert_approx_eq::<FloatElem>(&s1.into_data(), tolerance());
}

#[test]
fn test_svd_convergence_and_determinism() {
    // With enough sweeps the decomposition reconstructs the input, and two
    // runs over the same input return identical data (host pipeline is
    // deterministic; backend overrides must be too for this to hold).
    let device = Default::default();
    let tensor = TestTensor::<2>::random([12, 10], Distribution::Normal(0.0, 1.0), &device);
    let (u1, s1, vt1) = svd::<2, 1>(tensor.clone(), 30);
    assert_reconstruction::<2, 1>(&tensor, &u1, &s1, &vt1);
    let (u2, s2, vt2) = svd::<2, 1>(tensor, 30);
    assert_eq!(u1.into_data(), u2.into_data());
    assert_eq!(s1.into_data(), s2.into_data());
    assert_eq!(vt1.into_data(), vt2.into_data());
}

// ---------------------------------------------------------------------
// Dtype handling
// ---------------------------------------------------------------------

#[test]
fn test_svd_preserves_input_dtype() {
    // Inputs come back in the input dtype (half precision inputs are
    // computed in f32 internally, like `det`).
    let device = Default::default();
    let tensor = TestTensor::<3>::random([2, 4, 3], Distribution::Default, &device);
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_eq!(tensor.dtype(), u.dtype());
    assert_eq!(tensor.dtype(), s.dtype());
    assert_eq!(tensor.dtype(), vt.dtype());
    assert_reconstruction::<3, 2>(&tensor, &u, &s, &vt);
}

// ---------------------------------------------------------------------
// Empty inputs
// ---------------------------------------------------------------------

#[test]
fn test_svd_empty_matrices() {
    // Zero leading dimension: the reduced SVD is empty, no panic.
    let device = Default::default();
    let a = TestTensor::<2>::empty([0, 3], (&device, DType::F32));
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dims(), [0, 0]);
    assert_eq!(s.dims(), [0]);
    assert_eq!(vt.dims(), [0, 3]);

    let a = TestTensor::<2>::empty([3, 0], (&device, DType::F32));
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dims(), [3, 0]);
    assert_eq!(s.dims(), [0]);
    assert_eq!(vt.dims(), [0, 0]);
}

#[test]
fn test_svd_empty_batch_keeps_batch_dim() {
    let device = Default::default();
    let a = TestTensor::<3>::empty([0, 3, 4], (&device, DType::F32));
    let (u, s, vt) = svd::<3, 2>(a, 15);
    assert_eq!(u.dims(), [0, 3, 3]);
    assert_eq!(s.dims(), [0, 3]);
    assert_eq!(vt.dims(), [0, 3, 4]);
}

#[test]
fn test_svd_empty_matrices_with_batch_dim() {
    // Batched batches whose matrices have a zero dimension: the singular
    // value tensor keeps the full batch dimensions.
    let device = Default::default();
    let a = TestTensor::<3>::empty([2, 0, 3], (&device, DType::F32));
    let (u, s, vt) = svd::<3, 2>(a, 15);
    assert_eq!(u.dims(), [2, 0, 0]);
    assert_eq!(s.dims(), [2, 0]);
    assert_eq!(vt.dims(), [2, 0, 3]);

    let a = TestTensor::<3>::empty([2, 3, 0], (&device, DType::F32));
    let (u, s, vt) = svd::<3, 2>(a, 15);
    assert_eq!(u.dims(), [2, 3, 0]);
    assert_eq!(s.dims(), [2, 0]);
    assert_eq!(vt.dims(), [2, 0, 0]);
}

#[test]
fn test_svd_empty_f64_cast_back() {
    // The empty path must honor the f64 passthrough (no upcast happened,
    // but outputs are still produced in the original dtype).
    let device = Default::default();
    let a = TestTensor::<2>::zeros([2, 2], &device).cast(DType::F64);
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dtype(), DType::F64);
    assert_eq!(s.dtype(), DType::F64);
    assert_eq!(vt.dtype(), DType::F64);

    let a = TestTensor::<2>::empty([0, 2], (&device, DType::F64)).cast(DType::F64);
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dims(), [0, 0]);
    assert_eq!(u.dtype(), DType::F64);
    assert_eq!(s.dtype(), DType::F64);
    assert_eq!(vt.dtype(), DType::F64);
}

#[test]
fn test_svd_zero_column_basis() {
    // Zero m x 1 column: sigma is 0 and U stays a unit basis vector instead
    // of an arbitrary direction.
    let device = Default::default();
    let a = TestTensor::<2>::zeros([5, 1], &device);
    let (u, s, vt) = svd::<2, 1>(a, 15);
    let vals = singular_values(s);
    assert_eq!(vals[0], 0.0);
    let norm: f32 = u.powf_scalar(2.0).sum().into_scalar();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "U must stay orthonormal, got {norm}"
    );
    let vv = vt.to_data().convert::<f32>().to_vec::<f32>().unwrap();
    assert_eq!(vv[0], 1.0);
}

// ---------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------

#[test]
#[should_panic]
fn test_svd_panics_on_bad_generic_rank() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let _ = svd::<2, 2>(tensor, 10);
}

#[test]
#[should_panic]
fn test_svd_panics_on_1d_input() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _ = svd::<1, 0>(tensor, 10);
}
