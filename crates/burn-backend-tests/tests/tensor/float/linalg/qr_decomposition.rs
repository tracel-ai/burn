use super::*;
use burn_tensor::cast::ToElement;
use burn_tensor::{Tolerance, linalg, s};

fn assert_orthonormal(q: TestTensor<2>, tolerance: Tolerance<FloatElem>) {
    let device = q.device();
    let [_m, k] = q.dims();
    let eye = TestTensor::<2>::eye(k, &device);
    let qtq = q.clone().transpose().matmul(q);
    qtq.into_data()
        .assert_approx_eq::<FloatElem>(&eye.into_data(), tolerance);
}

// QR factors are unique up to column-wise sign flips; align to reference.
fn align_qr_to_expected(
    mut q: TestTensor<2>,
    mut r: TestTensor<2>,
    q_expected: TestTensor<2>,
) -> (TestTensor<2>, TestTensor<2>) {
    let [_m, k] = q_expected.dims();
    for col in 0..k {
        let q_col = q.clone().slice(s![.., col..(col + 1)]);
        let q_ref = q_expected.clone().slice(s![.., col..(col + 1)]);
        let dot = (q_col.clone() * q_ref).sum().into_scalar().to_f64();
        if dot < 0.0 {
            q = q.slice_assign(s![.., col..(col + 1)], -q_col);
            let r_row = r.clone().slice(s![col..(col + 1), ..]);
            r = r.slice_assign(s![col..(col + 1), ..], -r_row);
        }
    }
    (q, r)
}

#[test]
fn test_qr_square_reconstruction() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]],
        &device,
    );
    let (q, r) = linalg::qr_decomposition(tensor.clone());

    assert_eq!(q.dims(), [3, 3]);
    assert_eq!(r.dims(), [3, 3]);

    let reconstructed = q.clone().matmul(r.clone());
    let tolerance = Tolerance::permissive();
    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
    let q_expected = TestTensor::<2>::from_data(
        [
            [-0.85714287, 0.3942857, 0.33142856],
            [-0.42857143, -0.9028571, -0.034285713],
            [0.2857143, -0.17142858, 0.94285715],
        ],
        &device,
    );
    let r_expected = TestTensor::<2>::from_data(
        [[-14.0, -21.0, 14.0], [0.0, -175.0, 70.0], [0.0, 0.0, -35.0]],
        &device,
    );
    let (q_aligned, r_aligned) = align_qr_to_expected(q, r, q_expected.clone());
    q_aligned
        .clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&q_expected.into_data(), tolerance);
    r_aligned
        .into_data()
        .assert_approx_eq::<FloatElem>(&r_expected.into_data(), tolerance);
    assert_orthonormal(q_aligned, tolerance);
}

#[test]
fn test_qr_tall_reconstruction() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &device);
    let (q, r) = linalg::qr_decomposition(tensor.clone());

    assert_eq!(q.dims(), [4, 2]);
    assert_eq!(r.dims(), [2, 2]);

    let reconstructed = q.clone().matmul(r.clone());
    let tolerance = Tolerance::permissive();
    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
    let q_expected = TestTensor::<2>::from_data(
        [
            [-0.10910895, -0.82951504],
            [-0.32732683, -0.43915504],
            [-0.54554474, -0.048795003],
            [-0.7637626, 0.341565],
        ],
        &device,
    );
    let r_expected =
        TestTensor::<2>::from_data([[-9.165152, -10.910894], [0.0, -0.97590005]], &device);
    let (q_aligned, r_aligned) = align_qr_to_expected(q, r, q_expected.clone());
    q_aligned
        .clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&q_expected.into_data(), tolerance);
    r_aligned
        .into_data()
        .assert_approx_eq::<FloatElem>(&r_expected.into_data(), tolerance);
    assert_orthonormal(q_aligned, tolerance);
}

#[test]
fn test_qr_wide_reconstruction() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device);
    let (q, r) = linalg::qr_decomposition(tensor.clone());

    assert_eq!(q.dims(), [2, 2]);
    assert_eq!(r.dims(), [2, 4]);

    let reconstructed = q.clone().matmul(r.clone());
    let tolerance = Tolerance::permissive();
    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
    let q_expected = TestTensor::<2>::from_data(
        [[-0.19611613, -0.9805807], [-0.9805807, 0.19611613]],
        &device,
    );
    let r_expected = TestTensor::<2>::from_data(
        [
            [-5.0990195, -6.2757163, -7.452413, -8.62911],
            [0.0, -0.78446454, -1.5689291, -2.3533936],
        ],
        &device,
    );
    let (q_aligned, r_aligned) = align_qr_to_expected(q, r, q_expected.clone());
    q_aligned
        .clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&q_expected.into_data(), tolerance);
    r_aligned
        .into_data()
        .assert_approx_eq::<FloatElem>(&r_expected.into_data(), tolerance);
    assert_orthonormal(q_aligned, tolerance);
}
