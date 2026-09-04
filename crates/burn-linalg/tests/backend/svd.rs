use super::*;
use burn_core::tensor::TensorData;
use burn_core::tensor::{DType, Distribution, Tolerance};
use burn_linalg::svd;

fn tolerance() -> Tolerance<FloatElem> {
    Tolerance::rel_abs(5e-4, 1e-5).set_half_precision_absolute(5e-2)
}

/// Singular values read back as plain scalars (ordering and sign checks).
fn singular_values(s: TestTensor<1>) -> Vec<f32> {
    s.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
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
fn test_svd_boundary_shapes() {
    let device = Default::default();

    let tensor = TestTensor::<2>::from_data([[-7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!((u.dims(), s.dims(), vt.dims()), ([1, 1], [1], [1, 1]));
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
    s.into_data().assert_approx_eq::<FloatElem>(
        &TestTensor::<1>::from_data([7.0], &device).into_data(),
        tolerance(),
    );

    let tensor = TestTensor::<2>::from_data([[3.0, 4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!((u.dims(), s.dims(), vt.dims()), ([1, 1], [1], [1, 2]));
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);

    let tensor = TestTensor::<2>::from_data([[3.0], [4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!((u.dims(), s.dims(), vt.dims()), ([2, 1], [1], [1, 1]));
    assert_reconstruction::<2, 1>(&tensor, &u, &s, &vt);
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

// ---------------------------------------------------------------------
// Orthonormality of the factors
// ---------------------------------------------------------------------

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
fn test_svd_clustered_values() {
    // Near-equal singular values exercise the shifted QR convergence path.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[2.0, 0.05], [0.05, 1.95]], &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let expected = TestTensor::<1>::from_data([2.030_901_7, 1.919_098_3], &device);
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
fn test_svd_empty_shapes_and_dtype() {
    // Empty matrices preserve every batch dimension and the input dtype.
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

    let a = TestTensor::<3>::empty([0, 3, 4], (&device, DType::F32));
    let (u, s, vt) = svd::<3, 2>(a, 15);
    assert_eq!(u.dims(), [0, 3, 3]);
    assert_eq!(s.dims(), [0, 3]);
    assert_eq!(vt.dims(), [0, 3, 4]);

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

    let a = TestTensor::<2>::empty([0, 2], (&device, DType::F64)).cast(DType::F64);
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dims(), [0, 0]);
    assert_eq!(u.dtype(), DType::F64);
    assert_eq!(s.dtype(), DType::F64);
    assert_eq!(vt.dtype(), DType::F64);
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
#[should_panic(expected = "input tensor for SVD decomposition")]
fn test_svd_panics_on_1d_input() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _ = svd::<1, 0>(tensor, 10);
}

#[test]
#[should_panic(expected = "sweeps must be greater than zero")]
fn test_svd_panics_on_zero_sweeps() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let _ = svd::<2, 1>(tensor, 0);
}
