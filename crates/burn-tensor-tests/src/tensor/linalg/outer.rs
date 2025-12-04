use crate::*;
use burn_tensor::{ElementConversion, Tolerance};
use burn_tensor::{TensorData, linalg};

// ---------- Vector (D=1, R=2) tests ----------

#[test]
fn test_outer_basic() {
    let u = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let v = TestTensor::<1>::from([4.0, 5.0]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_shapes_only() {
    let device = Default::default();
    let u = TestTensor::<1>::zeros([3], &device);
    let v = TestTensor::<1>::zeros([5], &device);
    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v);
    assert_eq!(out.shape().dims(), [3, 5]);
}

#[test]
fn test_outer_asymmetry_and_shapes() {
    let u = TestTensor::<1>::from([1.0, 2.0]);
    let v = TestTensor::<1>::from([3.0, 4.0, 5.0]);

    let uv = linalg::outer::<TestBackend, 1, 2, _>(u.clone(), v.clone());
    let vu = linalg::outer::<TestBackend, 1, 2, _>(v, u);

    assert_eq!(uv.shape().dims(), [2, 3]);
    assert_eq!(vu.shape().dims(), [3, 2]);
}

#[test]
fn test_outer_zero_left() {
    let device = Default::default();
    let u = TestTensor::<1>::zeros([3], &device);
    let v = TestTensor::<1>::from([7.0, 8.0]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::zeros::<FloatElem, _>([3, 2]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_zero_right() {
    let device = Default::default();
    let u = TestTensor::<1>::from([1.0, -2.0, 3.0]);
    let v = TestTensor::<1>::zeros([4], &device);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::zeros::<FloatElem, _>([3, 4]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_signs() {
    let u = TestTensor::<1>::from([-1.0, 2.0]);
    let v = TestTensor::<1>::from([3.0, -4.0]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[-3.0, 4.0], [6.0, -8.0]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_integer_inputs() {
    let u = TestTensorInt::<1>::from([1, 2, 3]);
    let v = TestTensorInt::<1>::from([4, 5]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[4, 5], [8, 10], [12, 15]]);

    out.assert_eq(&expected, false);
}

#[test]
fn test_outer_equivalence_to_matmul() {
    let u = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let v = TestTensor::<1>::from([4.0, 5.0]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u.clone(), v.clone()).into_data();

    let u2 = u.reshape([3, 1]);
    let v2 = v.reshape([1, 2]);
    let out_matmul = u2.matmul(v2).into_data();

    out.assert_approx_eq::<FloatElem>(&out_matmul, Tolerance::default());
}

#[test]
fn test_outer_vector_identity_right_mult() {
    let u = TestTensor::<1>::from([2.0, -1.0]);
    let v = TestTensor::<1>::from([3.0, 4.0]);
    let w = TestTensor::<1>::from([5.0, 6.0]);

    let uv = linalg::outer::<TestBackend, 1, 2, _>(u.clone(), v.clone());
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

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[12.0, 15.0, 18.0]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_large_values() {
    let big = 1.0e10;
    let u = TestTensor::<1>::from([big, -big]);
    let v = TestTensor::<1>::from([big, big]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();
    let expected = TensorData::from([[big * big, big * big], [-big * big, -big * big]]);

    let tol = Tolerance::relative(1e-6).set_half_precision_relative(1e-3);
    out.assert_approx_eq::<FloatElem>(&expected, tol);
}

#[test]
fn test_outer_nan_propagation() {
    let u = TestTensor::<1>::from([f32::NAN, 2.0]);
    let v = TestTensor::<1>::from([3.0, 4.0]);

    let out = linalg::outer::<TestBackend, 1, 2, _>(u, v).into_data();

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
    let out = linalg::outer::<TestBackend, 2, 3, _>(x, y).into_data();

    let expected = TensorData::from([
        [[5.0, 6.0, 7.0], [10.0, 12.0, 14.0]],
        [[24.0, 27.0, 30.0], [32.0, 36.0, 40.0]],
    ]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_batched_shapes() {
    let device = Default::default();
    let x = TestTensor::<2>::zeros([3, 4], &device);
    let y = TestTensor::<2>::zeros([3, 5], &device);
    let out = linalg::outer::<TestBackend, 2, 3, _>(x, y);
    assert_eq!(out.shape().dims(), [3, 4, 5]);
}

#[test]
fn test_outer_batched_zero_left() {
    let device = Default::default();
    let x = TestTensor::<2>::zeros([2, 3], &device);
    let y = TestTensor::<2>::from([[7.0, 8.0], [9.0, 10.0]]);
    let out = linalg::outer::<TestBackend, 2, 3, _>(x, y).into_data();

    let expected = TensorData::zeros::<FloatElem, _>([2, 3, 2]);
    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_batched_zero_right() {
    let device = Default::default();
    let x = TestTensor::<2>::from([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]);
    let y = TestTensor::<2>::zeros([2, 4], &device);
    let out = linalg::outer::<TestBackend, 2, 3, _>(x, y).into_data();

    let expected = TensorData::zeros::<FloatElem, _>([2, 3, 4]);
    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_batched_signs() {
    let x = TestTensor::<2>::from([[-1.0, 2.0], [3.0, -4.0]]);
    let y = TestTensor::<2>::from([[3.0, -4.0], [-5.0, 6.0]]);
    let out = linalg::outer::<TestBackend, 2, 3, _>(x, y).into_data();

    let expected = TensorData::from([[[-3.0, 4.0], [6.0, -8.0]], [[-15.0, 18.0], [20.0, -24.0]]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_outer_batched_equivalence_to_per_sample_outer() {
    let x = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let y = TestTensor::<2>::from([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]);
    let batched = linalg::outer::<TestBackend, 2, 3, _>(x.clone(), y.clone());

    for b in 0..2 {
        let idx = TestTensorInt::<1>::from([b as i32]);

        let xb2d = x.clone().select(0, idx.clone()); // (1, m)
        let yb2d = y.clone().select(0, idx); // (1, n)

        let dims_x: [usize; 2] = xb2d.shape().dims();
        let dims_y: [usize; 2] = yb2d.shape().dims();
        let (m, n) = (dims_x[1], dims_y[1]);

        let per = linalg::outer::<TestBackend, 1, 2, _>(xb2d.reshape([m]), yb2d.reshape([n]));

        let bat3d = batched
            .clone()
            .select(0, TestTensorInt::<1>::from([b as i32])); // (m, n)

        let per_len = per.shape().num_elements();
        let per_flat = per.reshape([per_len]).into_data();

        let bat_len = bat3d.shape().num_elements();
        let bat_flat = bat3d.reshape([bat_len]).into_data();

        bat_flat.assert_approx_eq::<FloatElem>(&per_flat, Tolerance::default());
    }
}

#[test]
#[should_panic]
fn test_outer_batched_mismatched_batches_panics() {
    let device = Default::default();
    let x = TestTensor::<2>::zeros([2, 3], &device);
    let y = TestTensor::<2>::zeros([3, 4], &device);
    let _ = linalg::outer::<TestBackend, 2, 3, _>(x, y);
}

#[test]
fn test_outer_dim() {
    let u = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let v = TestTensor::<2>::from([[4.0, 5.0], [5.0, 6.0]]);

    let out = linalg::outer_dim::<TestBackend, 2, 3, _, _>(u, v, 0).into_data();
    let expected = TensorData::from([[[4.0, 10.0], [5.0, 12.0]], [[12.0, 20.0], [15.0, 24.0]]]);

    out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
