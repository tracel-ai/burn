use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_abs_flipped() {
    // [1, -2, 3, -4] flipped -> [-4, 3, -2, 1]; abs -> [4, 3, 2, 1]
    let tensor = TestTensor::<1>::from([1.0, -2.0, 3.0, -4.0]);
    let flipped = tensor.flip([0]);

    let output = flipped.abs();
    let expected = TensorData::from([4.0, 3.0, 2.0, 1.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_abs_ops_float() {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

    let output = tensor.abs();

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), false);
}

#[test]
fn should_preserve_order_for_shared_selected_tensor() {
    let device = Default::default();
    let values = (0..64).map(|value| value as f32).collect::<Vec<_>>();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [4, 16]), &device);
    let indices = TestTensorInt::<1>::from_data([0, 2, 4, 6, 8, 10, 12, 14], &device);
    let selected = tensor.select(1, indices);
    let expected = selected.clone();

    let output = selected.abs();

    output.into_data().assert_eq(&expected.into_data(), false);
}
