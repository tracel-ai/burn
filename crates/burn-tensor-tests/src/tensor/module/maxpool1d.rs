use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::{max_pool1d, max_pool1d_with_indices};

#[test]
fn test_max_pool1d_simple() {
    let kernel_size = 3;
    let padding = 0;
    let stride = 1;
    let dilation = 1;

    let x = TestTensor::from([[
        [0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221],
        [0.8148, 0.5474, 0.9490, 0.7890, 0.5537, 0.5689],
    ]]);
    let y = TestTensor::<3>::from([[
        [0.9861, 0.5474, 0.4477, 0.8221],
        [0.949, 0.949, 0.949, 0.789],
    ]]);

    let output = max_pool1d(x, kernel_size, stride, padding, dilation);

    y.to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_different_padding_stride_kernel() {
    let kernel_size = 3;
    let padding = 1;
    let stride = 2;
    let dilation = 1;

    let x = TestTensor::from([[[0.6309, 0.6112, 0.6998, 0.4708]]]);
    let y = TestTensor::<3>::from([[[0.6309, 0.6998]]]);

    let output = max_pool1d(x, kernel_size, stride, padding, dilation);

    y.to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_with_neg() {
    let kernel_size = 3;
    let padding = 1;
    let stride = 1;
    let dilation = 1;

    let x = TestTensor::from([[[-0.6309, -0.6112, -0.6998, -0.4708]]]);
    let y = TestTensor::<3>::from([[[-0.6112, -0.6112, -0.4708, -0.4708]]]);

    let output = max_pool1d(x, kernel_size, stride, padding, dilation);

    y.to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_with_dilation() {
    let kernel_size = 2;
    let padding = 1;
    let stride = 1;
    let dilation = 2;

    let x = TestTensor::from([[
        [0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221],
        [0.8148, 0.5474, 0.9490, 0.7890, 0.5537, 0.5689],
    ]]);
    let y = TestTensor::<3>::from([[
        [0.5474, 0.9861, 0.5474, 0.4477, 0.8221, 0.3548],
        [0.5474, 0.9490, 0.7890, 0.9490, 0.7890, 0.5537],
    ]]);

    let output = max_pool1d(x, kernel_size, stride, padding, dilation);

    y.to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_with_indices() {
    let kernel_size = 2;
    let padding = 0;
    let stride = 1;
    let dilation = 1;

    let x = TestTensor::from([[[0.2479, 0.6386, 0.3166, 0.5742]]]);
    let indices = TensorData::from([[[1, 1, 3]]]);
    let y = TestTensor::<3>::from([[[0.6386, 0.6386, 0.5742]]]);

    let (output, output_indices) =
        max_pool1d_with_indices(x, kernel_size, stride, padding, dilation);

    y.to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    output_indices.into_data().assert_eq(&indices, false);
}

#[test]
fn test_max_pool1d_complex() {
    let kernel_size = 4;
    let padding = 2;
    let stride = 1;
    let dilation = 1;

    let x = TestTensor::from([[[0.5388, 0.0676, 0.7122, 0.8316, 0.0653]]]);
    let indices = TensorData::from([[[0, 2, 3, 3, 3, 3]]]);
    let y = TestTensor::<3>::from([[[0.5388, 0.7122, 0.8316, 0.8316, 0.8316, 0.8316]]]);

    let (output, output_indices) =
        max_pool1d_with_indices(x, kernel_size, stride, padding, dilation);

    y.to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    output_indices.into_data().assert_eq(&indices, false);
}
