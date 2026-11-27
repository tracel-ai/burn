use super::*;
use burn_tensor::TensorData;

#[test]
fn should_diff_matmul() {
    let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[11.0, 5.0], [11.0, 5.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[3.0, 3.0], [10.0, 10.0]]), false);
    tensor_3
        .to_data()
        .assert_eq(&TensorData::from([[18.0, 28.0], [14.0, 23.0]]), false);
}

#[test]
fn test_matmul_complex_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.matmul(tensor_3);

    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[44.0, 20.0], [44.0, 20.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[56.0, 56.0], [16.0, 16.0]]), false);
}

#[test]
fn test_matmul_complex_2() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.matmul(tensor_3.clone());
    let tensor_6 = tensor_1.clone().matmul(tensor_5);

    let grads = tensor_6.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[800.0, 792.0], [360.0, 592.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[264., 264.0], [344.0, 344.0]]), false);
}

#[test]
fn should_diff_matmul_broadcast_lhs_single_column() {
    // [1, 2, 3] @ [3, 3, 1]: the rhs gradient, lhsᵀ @ grad, is itself such a product.
    let device = AutodiffDevice::new();
    let lhs =
        TestTensor::<3>::from_data([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], &device).require_grad();
    let rhs = TestTensor::<3>::from_data(
        TensorData::new((0..9).map(|i| i as f32).collect(), [3, 3, 1]),
        &device,
    )
    .require_grad();
    let weights = TestTensor::<3>::from_data(
        TensorData::new((0..6).map(|i| i as f32).collect(), [3, 2, 1]),
        &device,
    );

    let output = lhs.clone().matmul(rhs.clone()).mul(weights).sum();
    let grads = output.backward();

    let grad_lhs = lhs.grad(&grads).unwrap();
    let grad_rhs = rhs.grad(&grads).unwrap();

    grad_lhs.to_data().assert_eq(
        &TensorData::from([[[30.0, 36.0, 42.0], [39.0, 48.0, 57.0]]]),
        false,
    );
    grad_rhs.to_data().assert_eq(
        &TensorData::from([
            [[4.0], [5.0], [6.0]],
            [[14.0], [19.0], [24.0]],
            [[24.0], [33.0], [42.0]],
        ]),
        false,
    );
}

/// The gradients of a `Flex32` matmul keep the dtype and match the `F32` ones.
#[test]
fn should_diff_matmul_flex32() {
    use burn_tensor::{DType, Distribution, FloatDType, Tolerance};

    let device = AutodiffDevice::new();
    if !device.supports_dtype(DType::Flex32) {
        return;
    }
    let lhs = TestTensor::<3>::random([2, 5, 6], Distribution::Uniform(-1., 1.), &device)
        .cast(FloatDType::F32)
        .into_data();
    let rhs = TestTensor::<3>::random([2, 6, 3], Distribution::Uniform(-1., 1.), &device)
        .cast(FloatDType::F32)
        .into_data();

    let grads = |dtype: FloatDType| {
        let lhs = TestTensor::<3>::from_data(lhs.clone(), &device)
            .cast(dtype)
            .require_grad();
        let rhs = TestTensor::<3>::from_data(rhs.clone(), &device)
            .cast(dtype)
            .require_grad();
        let grads = lhs.clone().matmul(rhs.clone()).sum().backward();
        [lhs.grad(&grads).unwrap(), rhs.grad(&grads).unwrap()]
    };

    let expected = grads(FloatDType::F32);
    for (grad, expected) in grads(FloatDType::Flex32).into_iter().zip(expected) {
        assert_eq!(grad.dtype(), DType::Flex32);
        grad.cast(FloatDType::F32)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-4, 1e-4));
    }
}
