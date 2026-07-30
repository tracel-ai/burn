use super::*;
use burn_tensor::linalg::svd;

#[test]
fn test_svd_rectangular() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        &device,
    );
    let (u, s, vt) = svd(a, 10);

    assert_eq!(u.dims(), [3, 2]);
    assert_eq!(s.dims(), [2]);
    assert_eq!(vt.dims(), [2, 2]);

    // U columns should be approximately orthonormal
    let utu = u.clone().transpose().matmul(u);
    let eye2 = TestTensor::<2>::eye(2, &device);
    let err = (utu - eye2).abs().max().into_scalar::<f32>();
    assert!(err < 0.5, "U^T U not ≈ I: {err}");
}

#[test]
fn test_svd_square() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        &device,
    );
    let (u, s, _vt) = svd(a, 10);
    assert_eq!(u.dims(), [3, 3]);
    assert_eq!(s.dims(), [3]);
}

#[test]
fn test_svd_singular_values_positive() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [[3.0, 0.0], [0.0, 1.0]],
        &device,
    );
    let (_u, s, _vt) = svd(a, 20);
    let vals: Vec<f32> = s.into_data().bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(vals[0] > vals[1], "singular values should be descending");
    assert!(vals[0] > 2.5, "first SV should be ~3.0, got {}", vals[0]);
}
