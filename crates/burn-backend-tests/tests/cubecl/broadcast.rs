use super::*;
use burn_tensor::Shape;

// A size-1 dimension broadcasts to the other operand's size, including 0. Regression for #5334,
// where broadcast_shape used the per-dimension maximum and turned a 0 into 1, panicking in debug
// and reading out of bounds in release.

#[test]
fn broadcast_add_zero_against_one_trailing() {
    let device = Default::default();
    let a = TestTensor::<2>::zeros([2, 0], &device);
    let b = TestTensor::<2>::zeros([2, 1], &device);
    let out = a + b;
    assert_eq!(out.shape(), Shape::new([2, 0]));
    // Materialize to exercise the kernel launch and read-back on the empty result.
    let _ = out.into_data();
}

#[test]
fn broadcast_add_one_against_zero_trailing() {
    let device = Default::default();
    let a = TestTensor::<2>::zeros([2, 1], &device);
    let b = TestTensor::<2>::zeros([2, 0], &device);
    assert_eq!((a + b).shape(), Shape::new([2, 0]));
}

#[test]
fn broadcast_add_zero_against_one_leading() {
    let device = Default::default();
    let a = TestTensor::<2>::zeros([0, 3], &device);
    let b = TestTensor::<2>::zeros([1, 3], &device);
    assert_eq!((a + b).shape(), Shape::new([0, 3]));
}

#[test]
fn broadcast_add_zero_against_zero() {
    let device = Default::default();
    let a = TestTensor::<2>::zeros([2, 0], &device);
    let b = TestTensor::<2>::zeros([2, 0], &device);
    assert_eq!((a + b).shape(), Shape::new([2, 0]));
}

#[test]
fn broadcast_comparison_zero_against_one() {
    let device = Default::default();
    let a = TestTensor::<2>::zeros([2, 0], &device);
    let b = TestTensor::<2>::zeros([2, 1], &device);
    assert_eq!(a.greater(b).shape(), Shape::new([2, 0]));
}

#[test]
fn broadcast_mask_where_zero_against_one() {
    let device = Default::default();
    let input = TestTensor::<2>::zeros([2, 0], &device);
    let mask = TestTensorBool::<2>::from_data([[true], [false]], &device);
    let value = TestTensor::<2>::zeros([2, 1], &device);
    assert_eq!(input.mask_where(mask, value).shape(), Shape::new([2, 0]));
}

// Ordinary broadcasting (1 -> N) must still produce the right shape and values.
#[test]
fn broadcast_add_regular_unchanged() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0], [2.0]], &device);
    let b = TestTensor::<2>::from_data([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], &device);
    let out = a + b;
    assert_eq!(out.shape(), Shape::new([2, 3]));
    out.into_data().assert_eq(
        &TestTensor::<2>::from_data([[11.0, 21.0, 31.0], [42.0, 52.0, 62.0]], &device).into_data(),
        false,
    );
}
