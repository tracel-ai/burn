use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn test_cov_1() {
    let data = TensorData::from([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.cov(1, 1);
    let expected =
        TensorData::from([[2.48917, -1.73333], [-1.73333, 15.33333]]).convert::<FloatElem>();

    let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_cov_4() {
    let data = TensorData::from([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.cov(1, 0);
    let expected = TensorData::from([[1.86687, -1.30000], [-1.30000, 11.5]]).convert::<FloatElem>();

    let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_cov_2() {
    let data = TensorData::from([[0.5, 1.8], [0.2, -2.0], [3.0, -4.0], [5.0, 0.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.cov(1, 1);
    let expected = TensorData::from([
        [0.845, -1.43, -4.55, -3.25],
        [-1.43, 2.42, 7.7, 5.5],
        [-4.55, 7.7, 24.5, 17.5],
        [-3.25, 5.5, 17.5, 12.5],
    ])
    .convert::<FloatElem>();

    let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_cov_3() {
    let data = TensorData::from([
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
    ]);
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(data, &device);
    let data_actual = tensor.cov(0, 1).into_data();
    let data_expected = TestTensor::<3>::zeros([4, 4, 4], &device).to_data();
    data_expected.assert_approx_eq::<FloatElem>(&data_actual, Tolerance::default());
}
