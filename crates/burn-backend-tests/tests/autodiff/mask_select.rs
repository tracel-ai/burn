use super::*;
use burn_tensor::TensorData;

#[test]
fn test_mask_select_grad() {
    // Gradients must flow back to the selected positions and be zero elsewhere.
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(
        TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        &device,
    )
    .require_grad();
    let mask = TestTensorBool::<2>::from_data(
        TensorData::from([[true, false, true], [false, true, false]]),
        &device,
    );

    // Selected values x = [0, 2, 4]; y = x * x, so dy/dx = 2x = [0, 4, 8].
    let tensor_2 = tensor_1.clone().mask_select(mask);
    let tensor_3 = tensor_2.clone() * tensor_2;

    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    grad_1
        .into_data()
        .assert_eq(&TensorData::from([[0., 0., 4.], [0., 8., 0.]]), false);
}

#[test]
fn test_mask_select_grad_empty_mask() {
    // An all-`false` mask must produce an all-zero gradient. The backward pass scatter-adds a
    // zero-length gradient into the input, exercising the zero-size select_add path.
    let device = AutodiffDevice::new();
    let tensor_1 =
        TestTensor::<1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device).require_grad();
    let mask = TestTensorBool::<1>::from_data(TensorData::from([false, false, false]), &device);

    let tensor_2 = tensor_1.clone().mask_select(mask);
    let tensor_3 = tensor_2.sum();

    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    grad_1
        .into_data()
        .assert_eq(&TensorData::from([0.0, 0.0, 0.0]), false);
}
