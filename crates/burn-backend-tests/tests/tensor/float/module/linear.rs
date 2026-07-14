use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::linear;

#[test]
fn test_linear_1d() {
    let weight = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let x = TestTensor::<1>::from([1.0, 2.0]);
    let output = linear(x, weight, None);

    let expected = TensorData::from([7.0, 10.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::relative(1e-5));
}

#[test]
fn test_linear_1d_one_element_output() {
    let weight = TestTensor::<2>::from([[3.0], [4.0]]);

    let x = TestTensor::<1>::from([1.0, 2.0]);
    let output = linear(x, weight, None);

    let expected = TensorData::from([11.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::relative(1e-5));
}

#[test]
fn test_linear_forward_no_bias() {
    let weight = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let x = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]);

    let output = linear(x, weight, None);

    let expected = TensorData::from([[[7.0, 10.0], [15.0, 22.0]], [[-7.0, -10.0], [-15.0, -22.0]]]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::relative(1e-5));
}

#[test]
fn test_linear_forward_with_bias() {
    let weight = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let bias = Some(TestTensor::<1>::from([1.0, -1.0]));

    let x = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]);

    let output = linear(x, weight, bias);

    let expected = TensorData::from([[[8.0, 9.0], [16.0, 21.0]], [[-6.0, -11.0], [-14.0, -23.0]]]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::relative(1e-5));
}
