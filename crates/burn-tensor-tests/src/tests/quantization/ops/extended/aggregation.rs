use crate::qtensor::*;
use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_should_mean() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.mean();

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([15.0 / 6.0]), Tolerance::absolute(1e-1));
}

#[test]
fn test_should_sum() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.sum();

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([15.0]), Tolerance::absolute(1e-1));
}

#[test]
fn test_should_mean_last_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.mean_dim(1);
    let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn test_should_sum_last_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.sum_dim(1);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[3.0], [12.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_should_sum_first_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

    let output = tensor.sum_dim(0);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[7.0, 3.0, 5.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_should_mean_first_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

    let output = tensor.mean_dim(0);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[7.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_should_sum_mid_dim_3d_non_contiguous_1() {
    let tensor = QTensor::<TestBackend, 3>::int8([
        [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
        [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
    ]);

    let output = tensor.swap_dims(0, 2).sum_dim(1);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::new(vec![9.0, 7.0, -1.0, 3.0, 4.0, 5.0], [3, 1, 2]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_should_sum_mid_dim_3d_non_contiguous_2() {
    let tensor = QTensor::<TestBackend, 3>::int8([
        [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
        [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
    ]);

    let output = tensor.swap_dims(0, 1).sum_dim(1);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::new(vec![5.0, 5.0, 3.0, 11.0, -3.0, 6.0], [2, 1, 3]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_prod_float() {
    let tensor = QTensor::<TestBackend, 2>::int8([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.prod();

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([240.0]), Tolerance::rel_abs(1e-1, 1e-1));

    let tensor_with_zero = QTensor::<TestBackend, 2>::int8([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
    let output = tensor_with_zero.prod();

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([0.0]), Tolerance::rel_abs(1e-1, 1e-1));
}

#[test]
fn test_prod_dim_float() {
    let tensor = QTensor::<TestBackend, 2>::int8([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.prod_dim(1);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[4.0], [60.0]]),
            Tolerance::absolute(1e-1),
        );

    let tensor_with_zero = QTensor::<TestBackend, 2>::int8([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
    let output = tensor_with_zero.prod_dim(1);
    let expected = TensorData::from([[0.0], [60.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
