use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_remainder() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data(
        TensorData::from([
            0.9742, 0.3676, 0.0905, 0.8066, 0.7072, 0.7883, 0.6987, 0.1560, 0.7179, 0.7874, 0.9032,
            0.1845,
        ]),
        &device,
    )
    .require_grad();
    let tensor_2 = TestAutodiffTensor::<1>::from_data(
        TensorData::from([
            0.3357, 0.0285, 0.4115, 0.5511, 0.8637, 0.3593, 0.3885, 0.2569, 0.0936, 0.7172, 0.4792,
            0.4898,
        ]),
        &device,
    )
    .require_grad();
    let tensor_3 = tensor_1.clone().remainder(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([
        -2.0, -12.0, -0.0, -1.0, -0.0, -2.0, -1.0, -0.0, -7.0, -1.0, -1.0, -0.0,
    ]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
