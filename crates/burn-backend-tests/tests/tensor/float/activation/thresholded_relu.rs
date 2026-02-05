use super::*;
use burn_tensor::{TensorData, activation};

#[test]
fn test_thresholded_relu_d2() {
    // alpha = 1.0 (ONNX default): x if x > 1.0, else 0
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 1.0, 0.5]]);

    let output = activation::thresholded_relu(tensor, 1.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]]), false);
}

#[test]
fn test_thresholded_relu_d2_alpha() {
    // alpha = 0.5: x if x > 0.5, else 0
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 0.5, 0.6]]);

    let output = activation::thresholded_relu(tensor, 0.5);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 0.0, 2.0], [3.0, 0.0, 0.6]]), false);
}
