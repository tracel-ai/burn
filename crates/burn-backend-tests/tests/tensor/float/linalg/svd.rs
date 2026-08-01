use super::*;
use burn_tensor::{Distribution, TensorData, Tolerance, linalg::svd};

const REL: f32 = 5e-3;
const ABS: f32 = 1e-3;
/// Tolerance for torch-reference values (LAPACK and Jacobi differ only in
/// floating-point rounding order; half precision gets a looser bound).
fn torch_tol() -> Tolerance<FloatElem> {
    Tolerance::rel_abs(5e-4, 5e-4).set_half_precision_absolute(2e-2)
}

/// Max abs error of `A - U diag(S) Vt` as an f32.
fn recon_err<const D: usize, const D1: usize>(
    a: TestTensor<D>,
    u: TestTensor<D>,
    s: &TestTensor<D1>,
    vt: TestTensor<D>,
) -> f32 {
    let recon = u.mul(s.clone().unsqueeze_dim(D - 2)).matmul(vt);
    (a - recon).abs().max().into_scalar::<f32>()
}

fn assert_reconstruction<const D: usize, const D1: usize>(
    a: TestTensor<D>,
    u: TestTensor<D>,
    s: &TestTensor<D1>,
    vt: TestTensor<D>,
) {
    let err = recon_err::<D, D1>(a, u, s, vt);
    assert!(err < ABS, "A != U S Vt, max abs error {err}");
}

/// Flip the sign of each row so its largest-magnitude entry is positive
/// (the sign freedom of the right singular vectors is per row of Vt).
fn align_row_signs(vals: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let (mut jmax, mut vmax) = (0usize, 0.0f32);
        for j in 0..cols {
            let v = vals[i * cols + j].abs();
            if v > vmax {
                vmax = v;
                jmax = j;
            }
        }
        if vals[i * cols + jmax] < 0.0 {
            for j in 0..cols {
                vals[i * cols + j] = -vals[i * cols + j];
            }
        }
    }
}

/// Flip the sign of each column so its largest-magnitude entry is positive.
/// SVD factor signs are arbitrary; this makes them comparable to reference
/// values. `vals` is row-major `[rows, cols]`.
fn align_column_signs(vals: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        let (mut imax, mut vmax) = (0usize, 0.0f32);
        for i in 0..rows {
            let v = vals[i * cols + j].abs();
            if v > vmax {
                vmax = v;
                imax = i;
            }
        }
        if vals[imax * cols + j] < 0.0 {
            for i in 0..rows {
                vals[i * cols + j] = -vals[i * cols + j];
            }
        }
    }
}

// ---------------------------------------------------------------------
// Reconstruction: A = U @ diag(S) @ Vt (the defining property)
// ---------------------------------------------------------------------

#[test]
fn test_svd_rectangular_reconstruction() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_square_reconstruction() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_wide_reconstruction() {
    // m < n: exercises the transpose path
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 2]);
    assert_eq!(s.dims(), [2]);
    assert_eq!(vt.dims(), [2, 3]);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_single_row() {
    // 1 x n: k = 1
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[3.0, 4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [1, 1]);
    assert_eq!(s.dims(), [1]);
    assert_eq!(vt.dims(), [1, 2]);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
    let sv: Vec<f32> = s
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!((sv[0] - 5.0).abs() < REL, "sigma ~ 5.0, got {}", sv[0]);
}

#[test]
fn test_svd_single_column() {
    // m x 1: k = 1
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[3.0], [4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 1]);
    assert_eq!(s.dims(), [1]);
    assert_eq!(vt.dims(), [1, 1]);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_1x1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[-7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
    let sv: Vec<f32> = s
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!((sv[0] - 7.0).abs() < REL, "sigma ~ 7.0, got {}", sv[0]);
}

#[test]
fn test_svd_batched_reconstruction() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        ],
        &device,
    );
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_eq!(u.dims(), [2, 3, 3]);
    assert_eq!(s.dims(), [2, 3]);
    assert_eq!(vt.dims(), [2, 3, 3]);
    assert_reconstruction::<3, 2>(tensor, u, &s, vt);
}

#[test]
fn test_svd_batched_mixed_matrices() {
    // Distinct matrices per batch element, including a rank-deficient one
    // (the null-space singular vectors differ arbitrarily; reconstruction
    // and singular values must still be exact).
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [[[1.0, 2.0], [2.0, 4.0]], [[1.0, 0.0], [0.0, 3.0]]],
        &device,
    );
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_reconstruction::<3, 2>(tensor, u, &s, vt);
    let vals: Vec<f32> = s
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // batch element 0: rank-1 [[1,2],[2,4]] -> sigma ~ [5, 0]
    assert!(
        (vals[0] - 5.0).abs() < REL,
        "batch 0 sigma1 ~ 5, got {}",
        vals[0]
    );
    assert!(vals[1].abs() < ABS, "batch 0 sigma2 ~ 0, got {}", vals[1]);
    // batch element 1: diag(1, 3) -> sigma = [3, 1]
    assert!(
        (vals[2] - 3.0).abs() < REL,
        "batch 1 sigma1 ~ 3, got {}",
        vals[2]
    );
    assert!(
        (vals[3] - 1.0).abs() < REL,
        "batch 1 sigma2 ~ 1, got {}",
        vals[3]
    );
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
    assert_reconstruction::<3, 2>(tensor, u, &s, vt);
}

// ---------------------------------------------------------------------
// Orthonormality of the factors
// ---------------------------------------------------------------------

#[test]
fn test_svd_orthonormal_factors() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor, 15);
    let _ = s;

    let utu = u.clone().transpose().matmul(u);
    let eye = TestTensor::<2>::eye(3, &device);
    let err = (utu - eye.clone()).abs().max().into_scalar::<f32>();
    assert!(err < REL, "U^T U != I, max abs error {err}");

    let vvt = vt.clone().matmul(vt.transpose());
    let err = (vvt - eye).abs().max().into_scalar::<f32>();
    assert!(err < REL, "Vt Vt^T != I, max abs error {err}");
}

#[test]
fn test_svd_orthonormal_rectangular() {
    // U [m, k] has orthonormal columns; Vt [k, n] has orthonormal rows.
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 4], Distribution::Normal(0.0, 1.0), &device);
    let (u, s, vt) = svd::<2, 1>(tensor, 15);
    let _ = s;
    let utu = u.clone().transpose().matmul(u);
    let eye = TestTensor::<2>::eye(4, &device);
    let err = (utu - eye.clone()).abs().max().into_scalar::<f32>();
    assert!(err < REL, "U^T U != I, max abs error {err}");
    let vvt = vt.clone().matmul(vt.transpose());
    let err = (vvt - eye).abs().max().into_scalar::<f32>();
    assert!(err < REL, "Vt Vt^T != I, max abs error {err}");
}

// ---------------------------------------------------------------------
// Singular values: known values, ordering, non-negativity
// ---------------------------------------------------------------------

#[test]
fn test_svd_diagonal_values() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[3.0, 0.0], [0.0, 1.0]], &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let vals: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(
        (vals[0] - 3.0).abs() < REL,
        "first SV ~ 3.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 1.0).abs() < REL,
        "second SV ~ 1.0, got {}",
        vals[1]
    );
}

#[test]
fn test_svd_identity_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 0.0], [0.0, 1.0]], &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor.clone(), 15);
    let vals: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!((vals[0] - 1.0).abs() < REL, "sigma1 ~ 1, got {}", vals[0]);
    assert!((vals[1] - 1.0).abs() < REL, "sigma2 ~ 1, got {}", vals[1]);
}

#[test]
fn test_svd_singular_values_descending() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 3], Distribution::Normal(0.0, 1.0), &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let vals: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for w in vals.windows(2) {
        assert!(
            w[0] >= w[1] - 1e-5,
            "singular values must be descending: {vals:?}"
        );
    }
    assert!(
        vals.iter().all(|&v| v >= 0.0),
        "must be non-negative: {vals:?}"
    );
}

#[test]
fn test_svd_singular_matrix() {
    // Rank-1 matrix: one nonzero singular value, second is ~0.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    let vals: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!((vals[0] - 5.0).abs() < REL, "sigma1 ~ 5.0, got {}", vals[0]);
    assert!(vals[1].abs() < ABS, "sigma2 ~ 0, got {}", vals[1]);
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_zero_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 0.0], [0.0, 0.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    // No NaNs anywhere; singular values are zero; reconstruction is exact.
    let vals: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(
        vals.iter().all(|v| v.is_finite() && v.abs() < ABS),
        "S must be ~0: {vals:?}"
    );
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_clustered_singular_values() {
    // Near-equal singular values are the hard case for power-iteration
    // schemes; Jacobi converges quadratically regardless.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[2.0, 0.05], [0.05, 1.95]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    let vals: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // sigma = eigenvalues of A (A is symmetric PSD) = (3.95 +- sqrt(0.0125)) / 2
    assert!(
        (vals[0] - 2.0309017).abs() < REL && (vals[1] - 1.9190983).abs() < REL,
        "clustered SVs wrong: {vals:?}"
    );
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

#[test]
fn test_svd_ill_conditioned() {
    // Hilbert 5x5: condition number ~ 5e5 - still reconstructs (relative).
    let device = Default::default();
    let flat: Vec<f32> = (0..25)
        .map(|idx| 1.0 / (idx / 5 + idx % 5 + 1) as f32)
        .collect();
    let tensor = TestTensor::<2>::from_data(TensorData::new(flat, [5, 5]), &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    let rel_err = recon_err::<2, 1>(tensor, u, &s, vt);
    let sv: Vec<f32> = s
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let scale = sv[0];
    assert!(
        rel_err < REL * scale + ABS,
        "Hilbert reconstruction err {rel_err} > tol {}",
        REL * scale + ABS
    );
}

#[test]
fn test_svd_scaling_invariance() {
    // sigma(2A) = 2 sigma(A); the singular vectors are unchanged.
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let (_u1, s1, _vt1) = svd::<2, 1>(a.clone(), 15);
    let (_u2, s2, _vt2) = svd::<2, 1>(a.mul_scalar(2.0), 15);
    let v1: Vec<f32> = s1
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let v2: Vec<f32> = s2
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(
        (v2[0] - 2.0 * v1[0]).abs() < REL && (v2[1] - 2.0 * v1[1]).abs() < REL,
        "sigma(2A) != 2 sigma(A): {v1:?} vs {v2:?}"
    );
}

#[test]
fn test_svd_transpose_equivalence() {
    // sigma(A) == sigma(A^T)
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let (_u1, s1, _vt1) = svd::<2, 1>(a.clone(), 15);
    let (_u2, s2, _vt2) = svd::<2, 1>(a.transpose(), 15);
    let v1: Vec<f32> = s1
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let v2: Vec<f32> = s2
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!(
            (a - b).abs() < REL,
            "sigma(A) != sigma(A^T): {v1:?} vs {v2:?}"
        );
    }
}

#[test]
fn test_svd_random_reconstruction() {
    // Random well-conditioned matrices reconstruct within tolerance.
    let device = Default::default();
    for _ in 0..3 {
        let tensor = TestTensor::<2>::random([6, 4], Distribution::Normal(0.0, 1.0), &device);
        let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
        assert_reconstruction::<2, 1>(tensor, u, &s, vt);
    }
}

// ---------------------------------------------------------------------
// Torch reference values (torch.linalg.svd, LAPACK gesdd).
// Factor signs are aligned column-wise before comparison (SVD signs are
// arbitrary); the singular values are compared exactly.
// ---------------------------------------------------------------------

#[test]
fn test_svd_torch_reference_rectangular() {
    // torch.linalg.svd([[1,2],[3,4],[5,6]]) - 3x2
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor, 15);
    let tol = torch_tol();

    let sv = TestTensor::<1>::from_data([9.52551937, 0.51430136], &device);
    s.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&sv.into_data(), tol);

    // torch u (row-major [3,2]) and vt ([2,2]), signs aligned per column.
    let mut uv: Vec<f32> = u
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut vv: Vec<f32> = vt
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    align_column_signs(&mut uv, 3, 2);
    align_row_signs(&mut vv, 2, 2);
    let ref_u = TestTensor::<2>::from_data(
        [
            [0.22984788, 0.88346142],
            [0.52474481, 0.24078178],
            [0.81964201, -0.40189564],
        ],
        &device,
    );
    let ref_vt = TestTensor::<2>::from_data(
        [[0.61962938, 0.78489453], [0.78489453, -0.61962938]],
        &device,
    );
    let got_u = TestTensor::<2>::from_data(TensorData::new(uv, [3, 2]), &device);
    let got_vt = TestTensor::<2>::from_data(TensorData::new(vv, [2, 2]), &device);
    got_u
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_u.into_data(), tol);
    got_vt
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_vt.into_data(), tol);
}

#[test]
fn test_svd_torch_reference_square() {
    // torch.linalg.svd([[4,7,3],[6,1,3],[8,3,7]]) - 3x3
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor, 15);
    let tol = torch_tol();

    let sv = TestTensor::<1>::from_data([14.67576408, 4.95769882, 1.42939591], &device);
    s.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&sv.into_data(), tol);

    let mut uv: Vec<f32> = u
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut vv: Vec<f32> = vt
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    align_column_signs(&mut uv, 3, 3);
    align_row_signs(&mut vv, 3, 3);
    let ref_u = TestTensor::<2>::from_data(
        [
            [0.51108384, 0.84870338, 0.13599968],
            [0.43565938, -0.39217851, 0.81018335],
            [0.74094146, -0.35482201, -0.57018161],
        ],
        &device,
    );
    let ref_vt = TestTensor::<2>::from_data(
        [
            [0.72131324, 0.42492306, 0.54694390],
            [-0.36243331, 0.90450847, -0.22473705],
            [-0.59021139, -0.03612483, 0.80644017],
        ],
        &device,
    );
    let got_u = TestTensor::<2>::from_data(TensorData::new(uv, [3, 3]), &device);
    let got_vt = TestTensor::<2>::from_data(TensorData::new(vv, [3, 3]), &device);
    got_u
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_u.into_data(), tol);
    got_vt
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_vt.into_data(), tol);
}

#[test]
fn test_svd_torch_reference_wide() {
    // torch.linalg.svd([[1,2,3],[4,5,6]]) - 2x3 (m < n transpose path)
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor, 15);
    let tol = torch_tol();

    let sv = TestTensor::<1>::from_data([9.50803089, 0.77286965], &device);
    s.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&sv.into_data(), tol);

    let mut uv: Vec<f32> = u
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut vv: Vec<f32> = vt
        .clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    align_column_signs(&mut uv, 2, 2);
    align_row_signs(&mut vv, 2, 3);
    let ref_u = TestTensor::<2>::from_data(
        [[0.38631779, 0.92236584], [0.92236584, -0.38631779]],
        &device,
    );
    let ref_vt = TestTensor::<2>::from_data(
        [
            [0.42866719, 0.56630695, 0.70394671],
            [0.80596405, 0.11238238, -0.58119905],
        ],
        &device,
    );
    let got_u = TestTensor::<2>::from_data(TensorData::new(uv, [2, 2]), &device);
    let got_vt = TestTensor::<2>::from_data(TensorData::new(vv, [2, 3]), &device);
    got_u
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_u.into_data(), tol);
    got_vt
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_vt.into_data(), tol);
}

#[test]
fn test_svd_torch_reference_singular() {
    // Rank-1 matrix: singular values match torch; the null-space factor is
    // arbitrary so only S (and reconstruction) are compared.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
    let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
    let sv = TestTensor::<1>::from_data([4.99999952, 0.00000011], &device);
    s.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&sv.into_data(), torch_tol());
    assert_reconstruction::<2, 1>(tensor, u, &s, vt);
}

// ---------------------------------------------------------------------
// Convergence and validation
// ---------------------------------------------------------------------

#[test]
fn test_svd_more_sweeps_improve_accuracy() {
    // Fixed 8x8 input (dense random draw, seed 42): on f32 CUDA backends a
    // fresh random draw occasionally keeps ~1e-3 of fused-reduction noise,
    // which makes an absolute bound flaky. A fixed matrix keeps the test
    // deterministic while still exercising the sweep-accuracy trade-off.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [
                0.3047, -1.0400, 0.7505, 0.9406, -1.9510, -1.3022, 0.1278, -0.3162,
            ],
            [
                -0.0168, -0.8530, 0.8794, 0.7778, 0.0660, 1.1272, 0.4675, -0.8593,
            ],
            [
                0.3688, -0.9589, 0.8785, -0.0499, -0.1849, -0.6809, 1.2225, -0.1545,
            ],
            [
                -0.4283, -0.3521, 0.5323, 0.3654, 0.4127, 0.4308, 2.1416, -0.4064,
            ],
            [
                -0.5122, -0.8138, 0.6160, 1.1290, -0.1139, -0.8402, -0.8245, 0.6506,
            ],
            [
                0.7433, 0.5432, -0.6655, 0.2322, 0.1167, 0.2187, 0.8714, 0.2236,
            ],
            [
                0.6789, 0.0676, 0.2891, 0.6313, -1.4572, -0.3197, -0.4704, -0.6389,
            ],
            [
                -0.2751, 1.4949, -0.8658, 0.9683, -1.6829, -0.3349, 0.1628, 0.5862,
            ],
        ],
        &device,
    );
    let (u3, s3, vt3) = svd::<2, 1>(tensor.clone(), 3);
    let (u30, s30, vt30) = svd::<2, 1>(tensor.clone(), 30);
    let err3 = recon_err::<2, 1>(tensor.clone(), u3, &s3, vt3);
    let err30 = recon_err::<2, 1>(tensor, u30, &s30, vt30);
    assert!(
        err3 > err30,
        "more sweeps must improve accuracy: {err3} vs {err30}"
    );
    assert!(err30 < ABS, "30 sweeps must converge, err {err30}");
}

#[test]
fn test_svd_f16_dtype_roundtrip() {
    // f16/bf16 inputs are upcast to f32 internally (like `det`); the factors
    // come back in the input dtype.
    let device = Default::default();
    let tensor = TestTensor::<3>::random([2, 3, 3], Distribution::Default, &device);
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 10);
    assert_eq!(tensor.dtype(), u.dtype());
    assert_eq!(tensor.dtype(), s.dtype());
    assert_eq!(tensor.dtype(), vt.dtype());
}

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

#[test]
#[ignore = "benchmark"]
fn bench_svd_vs_torch() {
    use std::time::Instant;
    let device = Default::default();
    let sizes = [
        (4usize, 4usize),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
    ];
    let reps = [100usize, 100, 50, 20, 10, 5];
    println!("size   | burn ndarray us | recon err");
    for ((m, n), r) in sizes.iter().zip(reps.iter()) {
        let a = TestTensor::<2>::random([*m, *n], Distribution::Normal(0.0, 1.0), &device);
        for _ in 0..(r / 5).max(1) {
            let _ = svd::<2, 1>(a.clone(), 15);
        }
        let t0 = Instant::now();
        for _ in 0..*r {
            let _ = svd::<2, 1>(a.clone(), 15);
        }
        let dt = t0.elapsed().as_secs_f64() / *r as f64;
        let (u, s, vt) = svd::<2, 1>(a.clone(), 15);
        let err = recon_err::<2, 1>(a, u, &s, vt);
        println!("{m}x{n}  | {:.1} | {err:.2e}", dt * 1e6);
    }
}
