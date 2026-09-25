use super::qtensor::*;
use super::*;
use burn_tensor::Distribution;

// `*_like` promises the input dtype, which can't be honored for a quantized input, and
// `one_hot_fill` would read lossy quantized values as class indices. Both reject it.

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn empty_like_should_panic_on_quantized() {
    let _ = QTensor::<1>::int8([1.0, 2.0, 3.0, 4.0]).empty_like();
}

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn zeros_like_should_panic_on_quantized() {
    let _ = QTensor::<1>::int8([1.0, 2.0, 3.0, 4.0]).zeros_like();
}

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn ones_like_should_panic_on_quantized() {
    let _ = QTensor::<1>::int8([1.0, 2.0, 3.0, 4.0]).ones_like();
}

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn full_like_should_panic_on_quantized() {
    let _ = QTensor::<1>::int8([1.0, 2.0, 3.0, 4.0]).full_like(5.0);
}

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn random_like_should_panic_on_quantized() {
    let _ = QTensor::<1>::int8([1.0, 2.0, 3.0, 4.0]).random_like(Distribution::Default);
}

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn one_hot_should_panic_on_quantized() {
    let _: TestTensor<2> = QTensor::<1>::int8([0.0, 1.0, 2.0, 3.0]).one_hot(4);
}

#[test]
#[should_panic(expected = "Quantized tensors are not supported")]
fn one_hot_fill_should_panic_on_quantized() {
    let _: TestTensor<2> = QTensor::<1>::int8([0.0, 1.0, 2.0, 3.0]).one_hot_fill(4, 1.0, 0.0, -1);
}
