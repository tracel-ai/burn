use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;
use burn_tensor::module::{max_pool1d, max_pool1d_with_indices};
use burn_tensor::ops::MaxPoolOptions;

#[test]
fn test_max_pool1d_simple() {
    let kernel_size = 4;
    let padding = 0;
    let stride = 1;
    let dilation = 1;

    let device = AutodiffDevice::new();
    let x = TestTensor::from_data(
        [[[0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestTensor::<3>::from_data([[[1., 1., 0., 0., 0., 1.]]], &device);

    let output = max_pool1d(
        x.clone(),
        MaxPoolOptions::new([kernel_size])
            .with_stride([stride])
            .with_padding([padding])
            .with_dilation([dilation]),
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_with_dilation() {
    let kernel_size = 4;
    let padding = 0;
    let stride = 1;
    let dilation = 2;

    let device = AutodiffDevice::new();
    let x = TestTensor::from_data(
        [[[
            0.5388, 0.0676, 0.7122, 0.8316, 0.0653, 0.9154, 0.1536, 0.9089, 0.8016, 0.7518, 0.2073,
            0.0501, 0.8811, 0.5604, 0.5075, 0.4384, 0.9963, 0.9698, 0.4988, 0.2609, 0.3391, 0.2230,
            0.4610, 0.5365, 0.6880,
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestTensor::<3>::from_data(
        [[[
            0., 0., 1., 0., 0., 3., 0., 1., 2., 1., 0., 0., 2., 0., 0., 0., 4., 4., 0., 0., 0., 0.,
            0., 0., 1.,
        ]]],
        &device,
    );

    let output = max_pool1d(
        x.clone(),
        MaxPoolOptions::new([kernel_size])
            .with_stride([stride])
            .with_padding([padding])
            .with_dilation([dilation]),
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_complex() {
    let kernel_size = 4;
    let padding = 0;
    let stride = 1;
    let dilation = 1;

    let device = AutodiffDevice::new();
    let x = TestTensor::from_data(
        [[[
            0.5388, 0.0676, 0.7122, 0.8316, 0.0653, 0.9154, 0.1536, 0.9089, 0.8016, 0.7518, 0.2073,
            0.0501, 0.8811, 0.5604, 0.5075, 0.4384, 0.9963, 0.9698, 0.4988, 0.2609, 0.3391, 0.2230,
            0.4610, 0.5365, 0.6880,
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestTensor::<3>::from_data(
        [[[
            0., 0., 0., 2., 0., 4., 0., 2., 1., 0., 0., 0., 4., 0., 0., 0., 4., 1., 1., 0., 0., 0.,
            1., 1., 1.,
        ]]],
        &device,
    );

    let output = max_pool1d(
        x.clone(),
        MaxPoolOptions::new([kernel_size])
            .with_stride([stride])
            .with_padding([padding])
            .with_dilation([dilation]),
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_complex_with_padding() {
    let kernel_size = 4;
    let padding = 2;
    let stride = 1;
    let dilation = 1;

    let device = AutodiffDevice::new();
    let x = TestTensor::from_data(
        [[[
            0.5388, 0.0676, 0.7122, 0.8316, 0.0653, 0.9154, 0.1536, 0.9089, 0.8016, 0.7518, 0.2073,
            0.0501, 0.8811, 0.5604, 0.5075, 0.4384, 0.9963, 0.9698, 0.4988, 0.2609, 0.3391, 0.2230,
            0.4610, 0.5365, 0.6880,
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestTensor::<3>::from_data(
        [[[
            1., 0., 1., 2., 0., 4., 0., 2., 1., 0., 0., 0., 4., 0., 0., 0., 4., 1., 1., 0., 0., 0.,
            1., 1., 3.,
        ]]],
        &device,
    );

    let output = max_pool1d(
        x.clone(),
        MaxPoolOptions::new([kernel_size])
            .with_stride([stride])
            .with_padding([padding])
            .with_dilation([dilation]),
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_with_indices_grads_overlapping_padded() {
    let test = MaxPool1dWithIndicesTestCase {
        length: 11,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        ceil_mode: false,
    };
    test.assert_grads_match_max_pool1d();
}

#[test]
fn test_max_pool1d_with_indices_grads_ceil_mode() {
    // (10 - 3) / 2 + 1: floor gives 4 windows, ceil gives 5.
    let test = MaxPool1dWithIndicesTestCase {
        length: 10,
        kernel_size: 3,
        stride: 2,
        padding: 0,
        dilation: 1,
        ceil_mode: true,
    };
    test.assert_grads_match_max_pool1d();
}

#[test]
fn test_max_pool1d_with_indices_grads_dilated() {
    let test = MaxPool1dWithIndicesTestCase {
        length: 11,
        kernel_size: 2,
        stride: 2,
        padding: 1,
        dilation: 2,
        ceil_mode: false,
    };
    test.assert_grads_match_max_pool1d();
}

/// `max_pool1d_with_indices` registers its own backward and must route gradients like
/// `max_pool1d`.
struct MaxPool1dWithIndicesTestCase {
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
}

impl MaxPool1dWithIndicesTestCase {
    fn assert_grads_match_max_pool1d(self) {
        let device = AutodiffDevice::new();
        let data = TestTensor::<3>::random([2, 3, self.length], Distribution::Default, &device)
            .into_data();
        let options = MaxPoolOptions::new([self.kernel_size])
            .with_stride([self.stride])
            .with_padding([self.padding])
            .with_dilation([self.dilation])
            .with_ceil_mode(self.ceil_mode);

        let grad = |with_indices: bool| {
            let x = TestTensor::<3>::from_data(data.clone(), &device).require_grad();
            let output = if with_indices {
                max_pool1d_with_indices(x.clone(), options.clone()).0
            } else {
                max_pool1d(x.clone(), options.clone())
            };
            let weights =
                TestTensorInt::<1>::arange(1..output.shape().num_elements() as i64 + 1, &device)
                    .float()
                    .reshape::<3, _>(output.shape());
            let grads = (output * weights).sum().backward();
            x.grad(&grads).unwrap().into_data()
        };
        grad(true).assert_eq(&grad(false), false);
    }
}
