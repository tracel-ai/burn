use super::*;
use burn_tensor::{DType, Distribution, Element, TensorData, Tolerance, linalg::svd};

const REL: f32 = 5e-3;
const ABS: f32 = 1e-3;
/// Absolute tolerance, with a looser bound for half precision (f16/bf16),
/// which only keeps ~3 significant digits.
fn abs_tol() -> f32 {
    if matches!(FloatElem::dtype(), DType::F16 | DType::BF16) {
        2e-2
    } else {
        ABS
    }
}
/// Tolerance for torch-reference values (LAPACK and Jacobi differ only in
/// floating-point rounding order; half precision gets a looser bound).
fn torch_tol() -> Tolerance<FloatElem> {
    Tolerance::rel_abs(5e-4, 5e-4).set_half_precision_absolute(2e-2)
}

/// Read tensor data as f32 scalars regardless of the float dtype (convert
/// casts; `to_vec::<f32>` would fail on non-f32 data).
fn to_f32_vec<const D: usize>(t: TestTensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

/// Max abs error of `A - U diag(S) Vt` as an f32, computed with host scalar
/// math: on fused CUDA the test harness's own matmul can hit the inaccurate
/// autotune kernels and smear the error for larger sizes.
fn recon_err<const D: usize, const D1: usize>(
    a: TestTensor<D>,
    u: TestTensor<D>,
    s: &TestTensor<D1>,
    vt: TestTensor<D>,
) -> f32 {
    let dims = a.dims();
    let batch: usize = dims[..(D - 2)].iter().product();
    let (m, n) = (dims[D - 2], dims[D - 1]);
    let k = s.dims()[D1 - 1];
    let a = to_f32_vec(a);
    let u = to_f32_vec(u);
    let s = to_f32_vec(s.clone());
    let vt = to_f32_vec(vt);
    let mut err = 0.0f32;
    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += u[(b * m + i) * k + t] * s[b * k + t] * vt[(b * k + t) * n + j];
                }
                err = err.max((a[(b * m + i) * n + j] - acc).abs());
            }
        }
    }
    err
}

/// Max abs error of `V^T V - I` (or `V V^T - I` for row factors), host math.
fn ortho_err(data: &[f32], rows: usize, cols: usize) -> f32 {
    let mut err = 0.0f32;
    for i in 0..cols {
        for j in 0..cols {
            let mut acc = 0.0f32;
            for t in 0..rows {
                acc += data[t * cols + i] * data[t * cols + j];
            }
            err = err.max((acc - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    err
}

fn assert_reconstruction<const D: usize, const D1: usize>(
    a: TestTensor<D>,
    u: TestTensor<D>,
    s: &TestTensor<D1>,
    vt: TestTensor<D>,
) {
    let err = recon_err::<D, D1>(a, u, s, vt);
    assert!(err < abs_tol(), "A != U S Vt, max abs error {err}");
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
    let sv: Vec<f32> = to_f32_vec(s);
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
    let sv: Vec<f32> = to_f32_vec(s);
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
    // and singular values must still be correct).
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [[[1.0, 2.0], [2.0, 4.0]], [[1.0, 0.0], [0.0, 3.0]]],
        &device,
    );
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 15);
    assert_reconstruction::<3, 2>(tensor, u, &s, vt);
    let vals: Vec<f32> = to_f32_vec(s);
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

    let uv = to_f32_vec(u);
    let vtv = to_f32_vec(vt);
    let err = ortho_err(&uv, 3, 3);
    assert!(err < REL, "U^T U != I, max abs error {err}");
    let err = ortho_err(&vtv, 3, 3);
    assert!(err < REL, "Vt Vt^T != I, max abs error {err}");
}

#[test]
fn test_svd_orthonormal_rectangular() {
    // U [m, k] has orthonormal columns; Vt [k, n] has orthonormal rows.
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 4], Distribution::Normal(0.0, 1.0), &device);
    let (u, s, vt) = svd::<2, 1>(tensor, 15);
    let _ = s;
    let uv = to_f32_vec(u);
    let vtv = to_f32_vec(vt);
    let err = ortho_err(&uv, 6, 4);
    assert!(err < REL, "U^T U != I, max abs error {err}");
    let err = ortho_err(&vtv, 4, 4);
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
    let vals: Vec<f32> = to_f32_vec(s.clone());
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
    let vals: Vec<f32> = to_f32_vec(s.clone());
    assert!((vals[0] - 1.0).abs() < REL, "sigma1 ~ 1, got {}", vals[0]);
    assert!((vals[1] - 1.0).abs() < REL, "sigma2 ~ 1, got {}", vals[1]);
}

#[test]
fn test_svd_singular_values_descending() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 3], Distribution::Normal(0.0, 1.0), &device);
    let (_u, s, _vt) = svd::<2, 1>(tensor, 15);
    let vals: Vec<f32> = to_f32_vec(s.clone());
    for w in vals.windows(2) {
        assert!(
            w[0] >= w[1] - abs_tol(),
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
    let vals: Vec<f32> = to_f32_vec(s.clone());
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
    let vals: Vec<f32> = to_f32_vec(s.clone());
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
    let vals: Vec<f32> = to_f32_vec(s.clone());
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
    let sv: Vec<f32> = to_f32_vec(s.clone());
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
    let v1: Vec<f32> = to_f32_vec(s1);
    let v2: Vec<f32> = to_f32_vec(s2);
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
    let v1: Vec<f32> = to_f32_vec(s1);
    let v2: Vec<f32> = to_f32_vec(s2);
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
// arbitrary); the singular values are compared with a tight tolerance.
// ---------------------------------------------------------------------

/// Compare U and Vt against torch references, aligning sign freedom per
/// column (U) and per row (Vt).
fn assert_torch_factors<const D: usize>(
    u: TestTensor<D>,
    vt: TestTensor<D>,
    ref_u: &[f32],
    ref_vt: &[f32],
    shape_u: [usize; D],
    shape_vt: [usize; D],
) {
    let device = u.device();
    let tol = torch_tol();
    let mut uv: Vec<f32> = to_f32_vec(u);
    let mut vv: Vec<f32> = to_f32_vec(vt);
    let (ru, cv) = (shape_u[D - 2], shape_u[D - 1]);
    let (rv, cw) = (shape_vt[D - 2], shape_vt[D - 1]);
    align_column_signs(&mut uv, ru, cv);
    align_row_signs(&mut vv, rv, cw);
    let got_u = TestTensor::<D>::from_data(TensorData::new(uv, shape_u), &device);
    let got_vt = TestTensor::<D>::from_data(TensorData::new(vv, shape_vt), &device);
    let ref_u = TestTensor::<D>::from_data(TensorData::new(ref_u.to_vec(), shape_u), &device);
    let ref_vt = TestTensor::<D>::from_data(TensorData::new(ref_vt.to_vec(), shape_vt), &device);
    got_u
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_u.into_data(), tol);
    got_vt
        .into_data()
        .assert_approx_eq::<FloatElem>(&ref_vt.into_data(), tol);
}

#[test]
fn test_svd_torch_reference() {
    let device = Default::default();
    // (matrix, expected singular values, ref u, ref vt, u shape, vt shape).
    // Values from torch.linalg.svd (LAPACK gesdd), signs aligned.
    type TorchCase = (
        TestTensor<2>,
        Vec<f32>,
        &'static [f32],
        &'static [f32],
        [usize; 2],
        [usize; 2],
    );
    let cases: Vec<TorchCase> = vec![
        (
            TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device),
            vec![9.525_519, 0.51430136],
            &[
                0.22984788,
                0.883_461_4,
                0.524_744_8,
                0.24078178,
                0.819_642,
                -0.40189564,
            ],
            &[0.619_629_4, 0.784_894_5, 0.784_894_5, -0.619_629_4],
            [3, 2],
            [2, 2],
        ),
        (
            TestTensor::<2>::from_data(
                [[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]],
                &device,
            ),
            vec![14.675_764, 4.957_699, 1.429_395_9],
            &[
                0.51108384,
                0.848_703_4,
                0.13599968,
                0.43565938,
                -0.392_178_5,
                0.81018335,
                0.74094146,
                -0.354_822,
                -0.570_181_6,
            ],
            &[
                0.72131324,
                0.42492306,
                0.546_943_9,
                -0.362_433_3,
                0.904_508_5,
                -0.22473705,
                -0.590_211_4,
                -0.03612483,
                0.806_440_2,
            ],
            [3, 3],
            [3, 3],
        ),
        (
            // wide: m < n transpose path
            TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device),
            vec![9.508_031, 0.77286965],
            &[0.386_317_8, 0.92236584, 0.92236584, -0.386_317_8],
            &[
                0.428_667_2,
                0.56630695,
                0.703_946_7,
                0.80596405,
                0.11238238,
                -0.58119905,
            ],
            [2, 2],
            [2, 3],
        ),
    ];
    for (tensor, sv_ref, ref_u, ref_vt, shape_u, shape_vt) in cases {
        let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
        let n_sv = sv_ref.len();
        let sv = TestTensor::<1>::from_data(burn_tensor::TensorData::new(sv_ref, [n_sv]), &device);
        s.clone()
            .into_data()
            .assert_approx_eq::<FloatElem>(&sv.into_data(), torch_tol());
        assert_torch_factors::<2>(u.clone(), vt.clone(), ref_u, ref_vt, shape_u, shape_vt);
        assert_reconstruction::<2, 1>(tensor, u, &s, vt);
    }

    // Rank-1 matrix: the null-space factor is arbitrary, so only S (and
    // reconstruction) are compared.
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
    // Fixed 16x16 input (dense random draw, seed 42). 1 sweep (16 QR steps)
    // must not converge yet, 30 sweeps must; the host pipeline is
    // deterministic so this is stable across backends and runs.
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [
                0.3047, -1.0400, 0.7505, 0.9406, -1.9510, -1.3022, 0.1278, -0.3162, -0.0168,
                -0.8530, 0.8794, 0.7778, 0.0660, 1.1272, 0.4675, -0.8593,
            ],
            [
                0.3688, -0.9589, 0.8785, -0.0499, -0.1849, -0.6809, 1.2225, -0.1545, -0.4283,
                -0.3521, 0.5323, 0.3654, 0.4127, 0.4308, 2.1416, -0.4064,
            ],
            [
                -0.5122, -0.8138, 0.6160, 1.1290, -0.1139, -0.8402, -0.8245, 0.6506, 0.7433,
                0.5432, -0.6655, 0.2322, 0.1167, 0.2187, 0.8714, 0.2236,
            ],
            [
                0.6789, 0.0676, 0.2891, 0.6313, -1.4572, -0.3197, -0.4704, -0.6389, -0.2751,
                1.4949, -0.8658, 0.9683, -1.6829, -0.3349, 0.1628, 0.5862,
            ],
            [
                0.7112, 0.7933, -0.3487, -0.4624, 0.8580, -0.1913, -1.2757, -1.1333, -0.9195,
                0.4972, 0.1424, 0.6905, -0.4273, 0.1585, 0.6256, -0.3093,
            ],
            [
                0.4568, -0.6619, -0.3631, -0.3817, -1.1958, 0.4870, -0.4694, 0.0125, 0.4807,
                0.4465, 0.6654, -0.0985, -0.4233, -0.0797, -1.6873, -1.4471,
            ],
            [
                -1.3227, -0.9972, 0.3998, -0.9055, -0.3782, 1.2992, -0.3563, 0.7375, -0.9336,
                -0.2054, -0.9500, -0.3390, 0.8403, -1.7273, 0.4344, 0.2377,
            ],
            [
                -0.5941, -1.4461, 0.0721, -0.5295, 0.2327, 0.0219, 1.6018, -0.2394, -1.0235,
                0.1793, 0.2200, 1.3592, 0.8351, 0.3569, 1.4633, -1.1888,
            ],
            [
                -0.6398, -0.9266, -0.3898, -1.3767, 0.6352, -0.2222, -1.4708, -1.0156, 0.3135,
                0.8381, 1.9967, 2.9139, 0.4144, -0.9895, -2.1320, 0.2677,
            ],
            [
                -0.8129, -0.4154, -0.6121, -0.1408, 1.0660, 0.1570, -0.1586, -1.0357, -1.6747,
                -0.4863, -0.0538, 1.7679, 0.1303, 0.9827, -0.4993, -1.1849,
            ],
            [
                -0.9651, -0.7252, 2.1285, -0.8214, 0.8385, -0.9029, 0.9316, 0.3850, -0.1566,
                -0.0408, -0.6548, 0.4461, -0.4550, -1.2256, -1.2779, 0.1726,
            ],
            [
                1.5791, 0.1600, -0.1186, 0.2858, 1.3060, 0.2194, -0.4109, 1.1063, 0.4288, 1.5358,
                0.1832, -1.2245, -1.3682, 1.6509, 1.7237, -0.1795,
            ],
            [
                -0.3832, 1.4614, -1.1070, -0.8947, 0.6433, -0.3946, -0.0051, -0.1634, 0.3376,
                1.4075, 0.0906, 0.6439, -2.0502, -0.0487, -0.8432, -1.2188,
            ],
            [
                -0.8782, -0.3341, 0.9159, -1.3264, 0.0306, -0.4842, -0.3277, 1.0028, 0.5381,
                1.3374, -0.1545, -0.6959, -0.2239, 0.2425, 0.1766, -1.0844,
            ],
            [
                0.0905, 0.2282, 2.5175, 1.8768, -0.8532, -0.2874, -1.4634, -0.5907, 0.3156, 1.2059,
                -0.7291, -0.6541, -2.1473, -0.1627, -1.0624, -0.5294,
            ],
            [
                -0.8769, -0.0943, -1.7577, -1.4670, 2.1292, -1.2874, -1.0968, 1.8369, 2.9051,
                -1.1716, -0.3682, 0.3416, 1.7287, -0.9869, -0.2453, 0.7773,
            ],
        ],
        &device,
    );
    let (u3, s3, vt3) = svd::<2, 1>(tensor.clone(), 1);
    let (u30, s30, vt30) = svd::<2, 1>(tensor.clone(), 30);
    let err3 = recon_err::<2, 1>(tensor.clone(), u3, &s3, vt3);
    let err30 = recon_err::<2, 1>(tensor, u30, &s30, vt30);
    assert!(
        err3 > err30,
        "more sweeps must improve accuracy: {err3} vs {err30}"
    );
    assert!(err30 < abs_tol(), "30 sweeps must converge, err {err30}");
}

#[test]
fn test_svd_f16_dtype_roundtrip() {
    // f16/bf16 inputs are upcast to f32 internally (like `det`); the factors
    // come back in the input dtype and must still reconstruct the input
    // within half-precision tolerance (the upcast -> compute -> cast-back
    // path must not lose more precision than the dtype itself carries).
    let device = Default::default();
    let tensor = TestTensor::<3>::random([2, 3, 3], Distribution::Default, &device);
    let (u, s, vt) = svd::<3, 2>(tensor.clone(), 10);
    assert_eq!(tensor.dtype(), u.dtype());
    assert_eq!(tensor.dtype(), s.dtype());
    assert_eq!(tensor.dtype(), vt.dtype());
    let err = recon_err::<3, 2>(tensor.clone(), u, &s, vt);
    assert!(
        err < abs_tol(),
        "half-precision reconstruction err {err} (dtype {:?})",
        tensor.dtype()
    );
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
    println!("size   | burn us | recon err");
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

#[test]
fn test_svd_empty_matrix() {
    // Zero leading dimension: empty reduced SVD, no panic.
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        burn_tensor::TensorData::new(Vec::<f32>::new(), [0, 3]),
        &device,
    );
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dims(), [0, 0]);
    assert_eq!(s.dims(), [0]);
    assert_eq!(vt.dims(), [0, 3]);
    let a = TestTensor::<2>::from_data(
        burn_tensor::TensorData::new(Vec::<f32>::new(), [3, 0]),
        &device,
    );
    let (u, s, vt) = svd::<2, 1>(a, 15);
    assert_eq!(u.dims(), [3, 0]);
    assert_eq!(s.dims(), [0]);
    assert_eq!(vt.dims(), [0, 0]);
}

#[test]
fn test_svd_empty_batch() {
    // Zero batch dimension with a non-empty matrix: factors keep the batch
    // dim (S must not collapse to [1, ..]).
    let device = Default::default();
    let a = TestTensor::<3>::from_data(
        burn_tensor::TensorData::new(Vec::<f32>::new(), [0, 3, 4]),
        &device,
    );
    let (u, s, vt) = svd::<3, 2>(a, 15);
    assert_eq!(u.dims(), [0, 3, 3]);
    assert_eq!(s.dims(), [0, 3]);
    assert_eq!(vt.dims(), [0, 3, 4]);
}

#[test]
fn test_svd_negative_det_2x2() {
    // det < 0 inputs must still reconstruct exactly (regression: a previous
    // handedness fix in the 2x2 closed form negated Vt row 1 without U
    // column 1, breaking every negative-determinant matrix).
    let device = Default::default();
    for a in [
        [[1.0f64, 2.0], [3.0, 4.0]],
        [[0.0f64, 1.0], [1.0, 0.0]],
        [[-3.0f64, 1.0], [2.0, -1.0]],
    ] {
        let tensor = TestTensor::<2>::from_data(a, &device);
        let (u, s, vt) = svd::<2, 1>(tensor.clone(), 15);
        assert_reconstruction::<2, 1>(tensor, u, &s, vt);
    }
}

#[test]
fn test_svd_zero_m1_orthonormal() {
    // Zero m x 1 matrix: U stays orthonormal (unit basis), Vt = [1].
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        burn_tensor::TensorData::new(vec![0.0f32; 5], [5, 1]),
        &device,
    );
    let (u, s, vt) = svd::<2, 1>(a, 15);
    let sv: Vec<f32> = to_f32_vec(s);
    assert_eq!(sv[0], 0.0);
    let uv: Vec<f32> = to_f32_vec(u);
    let norm: f32 = uv.iter().map(|x| x * x).sum();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "U column must be a unit basis, norm {norm}"
    );
    let vv: Vec<f32> = to_f32_vec(vt);
    assert_eq!(vv[0], 1.0);
}

#[test]
#[ignore = "stress"]
fn dbgt_stress() {
    use burn_tensor::Distribution;
    let device = Default::default();
    let rel = 2e-4f32;
    let mut failures = 0usize;

    let mut check = |name: &str, a: TestTensor<2>, _expect_recon: f32| {
        // All checks run as host scalar math: on fused CUDA, matmul in the
        // test harness itself hits the inaccurate autotune kernels, which
        // would smear the SVD error for large sizes.
        let (u, s, vt) = svd::<2, 1>(a.clone(), 30);
        let uv = u.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let sv = s.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let vtv = vt.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let av = a
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let (m, n, k) = (a.dims()[0], a.dims()[1], sv.len());
        let mut err = 0.0f32;
        let mut amax = 0.0f32;
        let mut fsum = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += uv[i * k + t] * sv[t] * vtv[t * n + j];
                }
                err = err.max((av[i * n + j] - acc).abs());
                amax = amax.max(av[i * n + j].abs());
                fsum += av[i * n + j] * av[i * n + j];
            }
        }
        let ssum: f32 = sv.iter().map(|x| x * x).sum();
        let mut utu_err = 0.0f32;
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0f32;
                for t in 0..m {
                    acc += uv[t * k + i] * uv[t * k + j];
                }
                utu_err = utu_err.max((acc - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        let frob_ok = if ssum.is_finite() && fsum.is_finite() {
            (ssum - fsum).abs() <= rel * fsum.max(1e-12)
        } else {
            true // f32 overflow in the test's own Frobenius computation
        };
        let ok = err <= rel * amax.max(1e-6) && frob_ok && utu_err <= rel;
        if !ok {
            failures += 1;
            println!(
                "STRESS FAIL {name}: recon {err:.3e} (amax {amax:.3e}) frob {:.3e} utu {utu_err:.3e}",
                (ssum - fsum).abs()
            );
        } else {
            println!("STRESS ok   {name}: recon {err:.3e} utu {utu_err:.3e}");
        }
    };

    let mk = |m: usize, n: usize| -> TestTensor<2> {
        TestTensor::<2>::random([m, n], Distribution::Normal(0.0, 1.0), &device)
    };

    check("rand 16x8", mk(16, 8), 0.0);
    check("rand 8x16", mk(8, 16), 0.0);
    check("rand 32x32", mk(32, 32), 0.0);
    check("rand 64x32", mk(64, 32), 0.0);
    check("rand 32x64", mk(32, 64), 0.0);
    check("rand 4x128", mk(4, 128), 0.0);
    check("rand 128x4", mk(128, 4), 0.0);
    check("rand 128x128", mk(128, 128), 0.0);
    check("rand 256x256", mk(256, 256), 0.0);

    // rank-deficient: A = B diag(sigma) C
    for (name, sig) in [
        ("rankdef 32", vec![2.0f32; 16]),
        ("rankdef scaled", vec![1000.0f32; 8]),
        ("cluster 32", vec![1.0f32; 32]),
    ] {
        let k = sig.len();
        let b = TestTensor::<2>::random([32, k], Distribution::Normal(0.0, 1.0), &device);
        let c = TestTensor::<2>::random([k, 32], Distribution::Normal(0.0, 1.0), &device);
        let sd = TestTensor::<2>::from_data(burn_tensor::TensorData::new(sig, [k, 1]), &device);
        let a = b.mul(sd.transpose()).matmul(c);
        check(name, a, 0.0);
    }

    // extreme scales
    let diag3 = |x: f32, y: f32, z: f32| {
        TestTensor::<2>::from_data(
            burn_tensor::TensorData::new(vec![x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z], [3, 3]),
            &device,
        )
    };
    check("scale 1e20", diag3(1e20, 1.0, 1.0), 0.0);
    check("scale 1e-30", diag3(1e-30, 1.0, 1.0), 0.0);
    check("scale mixed", diag3(1e10, 1e-10, 1.0), 0.0);
    check(
        "zeros 16x16",
        TestTensor::<2>::from_data(
            burn_tensor::TensorData::new(vec![0.0f32; 256], [16, 16]),
            &device,
        ),
        0.0,
    );

    // deterministic 256x256 (same matrix on every backend), sigma compared to torch f64
    let n = 256usize;
    let mut m256 = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            m256.push(
                (((i * 7919 + j * 104729) % 100000) as f32 / 100000.0 - 0.5) * 2.0
                    + (i as f32 - j as f32) * 0.001,
            );
        }
    }
    let a256 = TestTensor::<2>::from_data(burn_tensor::TensorData::new(m256, [n, n]), &device);
    let (u, s, vt) = svd::<2, 1>(a256.clone(), 30);
    let s2 = s.clone();
    let sv: Vec<f32> = s2.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    println!("STRESS sv256 {:?}", &sv[..16]);
    let err = recon_err::<2, 1>(a256.clone(), u, &s, vt);
    println!("STRESS 256det recon {err:.3e}");
    let _ = a256;

    // determinism: two runs byte-identical
    let a = mk(6, 4);
    let (u1, s1, vt1) = svd::<2, 1>(a.clone(), 30);
    let (u2, s2, vt2) = svd::<2, 1>(a, 30);
    assert_eq!(
        u1.clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap(),
        u2.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    );
    assert_eq!(
        s1.clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap(),
        s2.clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
    );
    assert_eq!(
        vt1.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
        vt2.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    );
    println!("STRESS determinism ok");
    assert_eq!(failures, 0, "{failures} stress failures");
}

#[test]
#[ignore = "benchmark"]
fn bench_svd_realistic() {
    use std::time::Instant;
    let device = Default::default();
    let cases: Vec<(String, Vec<usize>, usize)> = vec![
        ("tall 512x128".into(), vec![512, 128], 5),
        ("wide 128x512".into(), vec![128, 512], 5),
        ("tall 256x64".into(), vec![256, 64], 10),
        ("batch 8x64x64".into(), vec![8, 64, 64], 5),
        ("batch 16x128x128".into(), vec![16, 128, 128], 2),
        ("square 256x256".into(), vec![256, 256], 5),
    ];
    for (name, dims, r) in cases {
        if dims.len() == 3 {
            let a = TestTensor::<3>::random(dims.clone(), Distribution::Normal(0.0, 1.0), &device);
            let _ = svd::<3, 2>(a.clone(), 15);
            let t0 = Instant::now();
            for _ in 0..r {
                let _ = svd::<3, 2>(a.clone(), 15);
            }
            let dt = t0.elapsed().as_secs_f64() / r as f64;
            let (u, s, vt) = svd::<3, 2>(a.clone(), 15);
            let err = recon_err::<3, 2>(a, u, &s, vt);
            println!("BENCHR {name} | {:.1} us | err {err:.2e}", dt * 1e6);
        } else {
            let a = TestTensor::<2>::random(dims.clone(), Distribution::Normal(0.0, 1.0), &device);
            let _ = svd::<2, 1>(a.clone(), 15);
            let t0 = Instant::now();
            for _ in 0..r {
                let _ = svd::<2, 1>(a.clone(), 15);
            }
            let dt = t0.elapsed().as_secs_f64() / r as f64;
            let (u, s, vt) = svd::<2, 1>(a.clone(), 15);
            let err = recon_err::<2, 1>(a, u, &s, vt);
            println!("BENCHR {name} | {:.1} us | err {err:.2e}", dt * 1e6);
        }
    }
}

#[test]
#[ignore = "stress"]
fn dbgt_find_bad_tall() {
    use burn_tensor::Distribution;
    let device = Default::default();
    let mut worst = (0.0f32, 0usize);
    for trial in 0..100 {
        let a = TestTensor::<2>::random([512, 128], Distribution::Normal(0.0, 1.0), &device);
        let (u, s, vt) = svd::<2, 1>(a.clone(), 15);
        let err = recon_err::<2, 1>(a, u, &s, vt);
        if err > worst.0 {
            worst = (err, trial);
        }
    }
    println!("DBGT worst tall err {:.3e} trial {}", worst.0, worst.1);
    assert!(worst.0 < 1e-3, "worst tall err too big: {:.3e}", worst.0);
}
