use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_mean() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.mean().unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[3.5, 9.5], [3.5, 9.5]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[-0.75, -0.75], [3.0, 3.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_sum_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.sum().unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[14.0, 38.0], [14.0, 38.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[-3.0, -3.0], [12.0, 12.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_sum_2() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.clone().sum_dim(1);
    let tensor_5 = tensor_4.mul(tensor_3);

    let grads = tensor_5.sum().backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[494.0, 722.0], [2990.0, 4370.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[690.0, 690.0], [958.0, 958.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_mean_dim() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.mean_dim(1).unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[4.0, 36.0], [3.0, -17.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[9.0, 9.0], [35.5, 35.5]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_sum_dim() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.sum_dim(1).unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[8.0, 72.0], [6.0, -34.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[18.0, 18.0], [71.0, 71.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_prod() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([2.0, 3.0, 4.0]), &device).require_grad();

    let output = tensor.clone().prod();
    let grads = output.backward();
    let grad = tensor.grad(&grads).unwrap();

    // grad_i = prod(x) / x_i = [24/2, 24/3, 24/4]
    let expected = TensorData::from([12.0, 8.0, 6.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_prod_with_negatives() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([2.0, -3.0, 4.0]), &device).require_grad();

    let output = tensor.clone().prod();

    // The forward value keeps its sign. The previous default impl computed
    // exp(sum(log(x))), which returns NaN for any negative input.
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([-24.0]), Tolerance::default());

    let grads = output.backward();
    let grad = tensor.grad(&grads).unwrap();

    // grad_i = prod(x) / x_i = [-24/2, -24/-3, -24/4]
    let expected = TensorData::from([-12.0, 8.0, -6.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_prod_dim() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();

    let output = tensor.clone().prod_dim(1);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // Per row: grad_ij = prod(row_i) / x_ij
    let expected = TensorData::from([[6.0, 3.0, 2.0], [30.0, 24.0, 20.0]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_prod_dim_with_negatives() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();

    let output = tensor.clone().prod_dim(0);

    output.clone().to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[-4.0, -10.0, 18.0]]),
        Tolerance::default(),
    );

    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // Per column: grad_ij = prod(col_j) / x_ij
    let expected = TensorData::from([[-4.0, 5.0, 6.0], [1.0, -2.0, 3.0]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

// The following tests are ignored due to the same limitation as cumprod: the
// gradient divides by the input, which produces NaN when the input contains
// zeros. The true gradient at a zero is finite (the product of the other
// elements), but recovering it needs the zero-safe exclusive-cumulative-product
// algorithm tracked in https://github.com/tracel-ai/burn/issues/3864.

#[test]
#[ignore = "prod gradient with zeros not yet implemented - produces NaN due to division by zero"]
fn should_diff_prod_single_zero() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([2.0, 0.0, 3.0, 4.0]), &device).require_grad();

    let output = tensor.clone().prod();
    let grads = output.backward();
    let grad = tensor.grad(&grads).unwrap();

    // Only the zero slot has a non-zero gradient (product of the others = 24).
    let expected = TensorData::from([0.0, 24.0, 0.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[ignore = "prod gradient with zeros not yet implemented - produces NaN due to division by zero"]
fn should_diff_prod_multiple_zeros() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<1>::from_data(TensorData::from([2.0, 0.0, 3.0, 0.0, 5.0]), &device)
        .require_grad();

    let output = tensor.clone().prod();
    let grads = output.backward();
    let grad = tensor.grad(&grads).unwrap();

    // Every leave-one-out product still contains a zero, so the gradient is zero.
    let expected = TensorData::from([0.0, 0.0, 0.0, 0.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
