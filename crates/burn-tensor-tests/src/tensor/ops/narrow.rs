use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Shape, TensorData};

#[test]
fn test_narrow_1() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        &Default::default(),
    );

    let output = tensor.clone().narrow(0, 0, 2);
    let expected = TensorData::from([[1., 2., 3.], [4., 5., 6.]]);

    assert_eq!(output.shape(), Shape::from([2, 3]));
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_narrow_2() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        &Default::default(),
    );

    let output = tensor.clone().narrow(1, 1, 2);
    let expected = TensorData::from([[2., 3.], [5., 6.], [8., 9.]]);
    assert_eq!(output.shape(), Shape::from([3, 2]));
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_narrow_3() {
    let device = &Default::default();
    let shape = Shape::new([8, 8]);
    let tensor = TestTensorInt::arange(0..shape.num_elements() as i64, &device)
        .reshape::<2, _>(shape)
        .float();

    let output = tensor.clone().narrow(0, 3, 4);
    let expected = TensorData::from([
        [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
        [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
        [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
        [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
    ]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[should_panic]
fn test_narrow_invalid_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        &Default::default(),
    );

    let _output = tensor.narrow(2, 0, 2);
}

#[test]
#[should_panic]
fn test_narrow_invalid_start() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        &Default::default(),
    );

    let _output = tensor.narrow(0, 3, 2);
}

#[test]
#[should_panic]
fn test_narrow_invalid_zero_length() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        &Default::default(),
    );

    let _output = tensor.narrow(0, 1, 0);
}

#[test]
#[should_panic]
fn test_narrow_invalid_length() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        &Default::default(),
    );

    let _output = tensor.narrow(0, 0, 4);
}
