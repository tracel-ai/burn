use super::*;
use burn_core::tensor::TensorData;
use burn_core::tensor::{ElementConversion, Tolerance};
use burn_linalg as linalg;

// ---------- Vector (D=1, R=2) tests ----------

#[test]
fn test_outer_basic() {
    let u = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let v = TestTensor::<1>::from([4.0, 5.0]);

    let out = linalg::outer::<1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_integer_inputs() {
    let u = TestTensorInt::<1>::from([1, 2, 3]);
    let v = TestTensorInt::<1>::from([4, 5]);

    let out = linalg::outer::<1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[4, 5], [8, 10], [12, 15]]);

    out.assert_eq(&expected, false);
}

#[test]
fn test_outer_vector_identity_right_mult() {
    let u = TestTensor::<1>::from([2.0, -1.0]);
    let v = TestTensor::<1>::from([3.0, 4.0]);
    let w = TestTensor::<1>::from([5.0, 6.0]);

    let uv = linalg::outer::<1, 2, _>(u.clone(), v.clone());
    let left = uv.matmul(w.clone().reshape([2, 1])).reshape([2]);

    let v_dot_w = v.dot(w);
    let right = u * v_dot_w;

    left.into_data()
        .assert_approx_eq::<FloatElem>(&right.into_data(), Tolerance::default());
}

#[test]
fn test_outer_length_one_vectors() {
    let u = TestTensor::<1>::from([3.0]);
    let v = TestTensor::<1>::from([4.0, 5.0, 6.0]);

    let out = linalg::outer::<1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[12.0, 15.0, 18.0]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_large_values() {
    let big = 1.0e10;
    let u = TestTensor::<1>::from([big, -big]);
    let v = TestTensor::<1>::from([big, big]);

    let out = linalg::outer::<1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[big * big, big * big], [-big * big, -big * big]]);

    let tol = Tolerance::relative(1e-6).set_half_precision_relative(1e-3);
    out.assert_approx_eq::<FloatElem>(&expected, tol);
}

#[test]
fn test_outer_nan_propagation() {
    let u = TestTensor::<1>::from([f32::NAN, 2.0]);
    let v = TestTensor::<1>::from([3.0, 4.0]);

    let out = linalg::outer::<1, 2, _>(u, v).into_data();

    let s: &[FloatElem] = out
        .as_slice::<FloatElem>()
        .expect("outer nan_propagation: as_slice failed");

    assert!(s[0].is_nan());
    assert!(s[1].is_nan());
    assert_eq!(s[2], 6.0f32.elem::<FloatElem>());
    assert_eq!(s[3], 8.0f32.elem::<FloatElem>());
}

// ---------- Batched (D=2, R=3) tests ----------

#[test]
fn test_outer_batched_basic() {
    let x = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let y = TestTensor::<2>::from([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]);
    let out = linalg::outer::<2, 3, _>(x, y).into_data();

    let expected = TensorData::from([
        [[5.0, 6.0, 7.0], [10.0, 12.0, 14.0]],
        [[24.0, 27.0, 30.0], [32.0, 36.0, 40.0]],
    ]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[should_panic]
fn test_outer_batched_mismatched_batches_panics() {
    let device = Default::default();
    let x = TestTensor::<2>::zeros([2, 3], &device);
    let y = TestTensor::<2>::zeros([3, 4], &device);
    let _ = linalg::outer::<2, 3, _>(x, y);
}

#[test]
fn test_outer_dim() {
    let u = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let v = TestTensor::<2>::from([[4.0, 5.0], [5.0, 6.0]]);

    let out = linalg::outer_dim::<2, 3, _, _>(u, v, 0).into_data();
    let expected = TensorData::from([[[4.0, 10.0], [5.0, 12.0]], [[12.0, 20.0], [15.0, 24.0]]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
