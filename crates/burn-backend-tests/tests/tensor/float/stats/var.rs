use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_var() {
    let tensor = TestTensor::<2>::from_data(
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        &Default::default(),
    );

    let output = tensor.var(1);
    let expected = TensorData::from([[2.4892], [15.3333]]).convert::<FloatElem>();

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_var_mean() {
    let tensor = TestTensor::<2>::from_data(
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        &Default::default(),
    );

    let (var, mean) = tensor.var_mean(1);

    let var_expected = TensorData::from([[2.4892], [15.3333]]).convert::<FloatElem>();
    let mean_expected = TensorData::from([[0.125], [1.]]).convert::<FloatElem>();

    var.into_data()
        .assert_approx_eq::<FloatElem>(&var_expected, Tolerance::default());
    mean.into_data()
        .assert_approx_eq::<FloatElem>(&mean_expected, Tolerance::default());
}

#[test]
fn test_var_bias() {
    let tensor = TestTensor::<2>::from_data(
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        &Default::default(),
    );

    let output = tensor.var_bias(1);
    let expected = TensorData::from([[1.86688], [11.5]]).convert::<FloatElem>();

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_var_mean_bias() {
    let tensor = TestTensor::<2>::from_data(
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        &Default::default(),
    );

    let (var, mean) = tensor.var_mean_bias(1);

    let var_expected = TensorData::from([[1.86688], [11.5]]).convert::<FloatElem>();
    let mean_expected = TensorData::from([[0.125], [1.]]).convert::<FloatElem>();

    var.into_data()
        .assert_approx_eq::<FloatElem>(&var_expected, Tolerance::default());
    mean.into_data()
        .assert_approx_eq::<FloatElem>(&mean_expected, Tolerance::default());
}

#[test]
fn test_variance_negative_dim() {
    let tensor = TestTensor::<2>::from_data(
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        &Default::default(),
    );

    let var = tensor.clone().var(1).into_data();
    tensor
        .clone()
        .var(-1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&var, Tolerance::default());

    let var_bias = tensor.clone().var_bias(1).into_data();
    tensor
        .clone()
        .var_bias(-1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&var_bias, Tolerance::default());

    let (var, mean) = tensor.clone().var_mean(1);
    let (var_negative, mean_negative) = tensor.clone().var_mean(-1);
    var_negative
        .into_data()
        .assert_approx_eq::<FloatElem>(&var.into_data(), Tolerance::default());
    mean_negative
        .into_data()
        .assert_approx_eq::<FloatElem>(&mean.into_data(), Tolerance::default());

    let (var, mean) = tensor.clone().var_mean_bias(1);
    let (var_negative, mean_negative) = tensor.var_mean_bias(-1);
    var_negative
        .into_data()
        .assert_approx_eq::<FloatElem>(&var.into_data(), Tolerance::default());
    mean_negative
        .into_data()
        .assert_approx_eq::<FloatElem>(&mean.into_data(), Tolerance::default());
}
