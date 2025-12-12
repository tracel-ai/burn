use super::*;
use burn_tensor::TensorData;

#[test]
fn test_prod_float() {
    let tensor_1 = TestTensor::<2>::from([[-5.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor_1.prod();

    output
        .into_data()
        .assert_eq(&TensorData::from([-600.0]), false);
}

#[test]
fn test_prod_dim_2d() {
    let f = TestTensor::<2>::from([[-5.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    f.clone()
        .prod_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[-10.0], [60.0]]), false);

    f.clone()
        .prod_dim(-1)
        .into_data()
        .assert_eq(&TensorData::from([[-10.0], [60.0]]), false);
}

#[test]
fn test_prod_dims_2d() {
    let f = TestTensor::<2>::from([[-5.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    f.clone()
        .prod_dims(&[1])
        .into_data()
        .assert_eq(&TensorData::from([[-10.0], [60.0]]), false);

    f.clone()
        .prod_dims(&[-1])
        .into_data()
        .assert_eq(&TensorData::from([[-10.0], [60.0]]), false);

    f.clone()
        .prod_dims(&[0, 1])
        .into_data()
        .assert_eq(&TensorData::from([[-600.0]]), false);
}
