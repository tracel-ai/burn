use super::*;
use burn_tensor::linalg::qr_decomposition;

#[test]
fn test_qr_1x1() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[5.0]], &device);
    let (q, r) = qr_decomposition(a);

    assert_eq!(q.dims(), [1, 1]);
    assert_eq!(r.dims(), [1, 1]);

    let qtq = q.clone().transpose().matmul(q);
    let err = (qtq - TestTensor::<2>::eye(1, &device))
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(err < 1e-3, "Q^T Q error for 1x1: {err}");

    // R should contain the norm: |5.0| = 5.0
    let r_val = r.into_scalar::<f32>();
    assert!(
        (r_val - 5.0).abs() < 1e-3,
        "R[0,0] should be ~5.0, got {r_val}"
    );
}

#[test]
fn test_qr_square_orthonormal() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
        &device,
    );
    let (q, _r) = qr_decomposition(a);

    // Q^T Q should be close to identity.
    let qtq = q.clone().transpose().matmul(q);
    let eye = TestTensor::<2>::eye(3, &device);
    let err = (qtq - eye).abs().max().into_scalar::<f32>();
    assert!(err < 1e-1, "Q^T Q not close to I, max |err| = {err}");
}

#[test]
fn test_qr_reconstruct() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
        &device,
    );
    let (q, r) = qr_decomposition(a.clone());

    let recon = q.matmul(r);
    let err = (recon - a).abs().max().into_scalar::<f32>();
    assert!(err < 1e-1, "QR reconstruction error: {err}");
}

#[test]
fn test_qr_rectangular() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let (q, r) = qr_decomposition(a);

    assert_eq!(q.dims(), [3, 2]);
    assert_eq!(r.dims(), [2, 2]);

    // R should be upper triangular.
    let r_lo = r.clone().slice([1..2, 0..1]).into_scalar::<f32>();
    assert!(r_lo.abs() < 1e-5, "R[1,0] should be ~0, got {r_lo}");

    // Q^T Q ~= I_2
    let qtq = q.clone().transpose().matmul(q);
    let eye2 = TestTensor::<2>::eye(2, &device);
    let err = (qtq - eye2).abs().max().into_scalar::<f32>();
    assert!(err < 1e-1, "Q^T Q error: {err}");
}

#[test]
#[should_panic(expected = "m >= n")]
fn test_qr_wide_matrix_panics() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let _ = qr_decomposition(a);
}
