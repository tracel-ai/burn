use super::*;
use burn_tensor::{
    Shape, Tolerance,
    module::conv1d,
    ops::{ConvOptions, PaddedConvOptions},
};

#[test]
fn test_conv1d_basic() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        length: 4,
    };
    let device = AutodiffDevice::new();
    let grads = Grads {
        x: TestTensor::from_data(
            [
                [[14., 24., 24., 18.], [26., 42., 42., 30.]],
                [[14., 24., 24., 18.], [26., 42., 42., 30.]],
            ],
            &device,
        ),
        weight: TestTensor::from_data(
            [
                [[30., 44., 36.], [54., 76., 60.]],
                [[30., 44., 36.], [54., 76., 60.]],
            ],
            &device,
        ),
        bias: TestTensor::from_data([8., 8.], &device),
    };
    test.assert_grads(grads);
}

#[test]
fn test_conv1d_different_channels() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 2,
        channels_out: 3,
        kernel_size: 3,
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        length: 4,
    };
    let device = AutodiffDevice::new();
    let grads = Grads {
        x: TestTensor::from_data(
            [
                [[39., 63., 63., 45.], [57., 90., 90., 63.]],
                [[39., 63., 63., 45.], [57., 90., 90., 63.]],
            ],
            &device,
        ),
        weight: TestTensor::from_data(
            [
                [[30., 44., 36.], [54., 76., 60.]],
                [[30., 44., 36.], [54., 76., 60.]],
                [[30., 44., 36.], [54., 76., 60.]],
            ],
            &device,
        ),
        bias: TestTensor::from_data([8., 8., 8.], &device),
    };
    test.assert_grads(grads);
}

#[test]
fn test_conv1d_with_padding() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 2,
        stride: 1,
        dilation: 1,
        groups: 1,
        length: 4,
    };
    let device = AutodiffDevice::new();
    let grads = Grads {
        x: TestTensor::from_data(
            [
                [[24., 24., 24., 24.], [42., 42., 42., 42.]],
                [[24., 24., 24., 24.], [42., 42., 42., 42.]],
            ],
            &device,
        ),
        weight: TestTensor::from_data(
            [
                [[44., 44., 44.], [76., 76., 76.]],
                [[44., 44., 44.], [76., 76., 76.]],
            ],
            &device,
        ),
        bias: TestTensor::from_data([12., 12.], &device),
    };
    test.assert_grads(grads);
}

#[test]
fn test_conv1d_with_stride() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        stride: 2,
        dilation: 1,
        groups: 1,
        length: 4,
    };
    let device = AutodiffDevice::new();
    let grads = Grads {
        x: TestTensor::from_data(
            [
                [[8., 16., 8., 10.], [14., 28., 14., 16.]],
                [[8., 16., 8., 10.], [14., 28., 14., 16.]],
            ],
            &device,
        ),
        weight: TestTensor::from_data(
            [
                [[10., 20., 24.], [18., 36., 40.]],
                [[10., 20., 24.], [18., 36., 40.]],
            ],
            &device,
        ),
        bias: TestTensor::from_data([4., 4.], &device),
    };
    test.assert_grads(grads);
}

#[test]
fn test_conv1d_dilation() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        stride: 1,
        dilation: 2,
        groups: 1,
        length: 4,
    };
    let device = AutodiffDevice::new();
    let grads = Grads {
        x: TestTensor::from_data(
            [
                [[6., 8., 8., 10.], [12., 14., 14., 16.]],
                [[6., 8., 8., 10.], [12., 14., 14., 16.]],
            ],
            &device,
        ),
        weight: TestTensor::from_data(
            [
                [[8., 22., 14.], [16., 38., 22.]],
                [[8., 22., 14.], [16., 38., 22.]],
            ],
            &device,
        ),
        bias: TestTensor::from_data([4., 4.], &device),
    };
    test.assert_grads(grads);
}

#[test]
fn test_conv1d_groups() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 2,
        length: 4,
    };
    let device = AutodiffDevice::new();
    let grads = Grads {
        x: TestTensor::from_data(
            [
                [[1., 3., 3., 3.], [7., 12., 12., 9.]],
                [[1., 3., 3., 3.], [7., 12., 12., 9.]],
            ],
            &device,
        ),
        weight: TestTensor::from_data([[[30., 44., 36.]], [[54., 76., 60.]]], &device),
        bias: TestTensor::from_data([8., 8.], &device),
    };
    test.assert_grads(grads);
}

/// Regression test for https://github.com/tracel-ai/burn/issues/4799.
///
/// When a conv drops more than one tail input (here length=11, kernel=4,
/// stride=4 → three dropped inputs), the transpose-conv used in the backward
/// pass must inject enough `padding_out` to reproduce the original input shape.
/// An earlier formula capped this at 1, which caused downstream ops (pad,
/// slice_assign) to panic on shape mismatch.
#[test]
fn test_conv1d_backward_shape_with_remainder() {
    let device = AutodiffDevice::new();
    let length = 11;
    let x = TestTensor::<3>::from_data(
        TestTensorInt::arange(0..(1 * 1 * length) as i64, &device)
            .reshape::<3, _>(Shape::new([1, 1, length]))
            .into_data(),
        &device,
    )
    .require_grad();
    let weight = TestTensor::<3>::from_data(
        TestTensorInt::arange(0..4, &device)
            .reshape::<3, _>(Shape::new([1, 1, 4]))
            .into_data(),
        &device,
    )
    .require_grad();

    let output = conv1d(
        x.clone(),
        weight.clone(),
        None,
        ConvOptions::new([4], [0], [1], 1),
    );
    let grads = output.sum().backward();

    let x_grad = x.grad(&grads).unwrap();
    let weight_grad = weight.grad(&grads).unwrap();
    assert_eq!(x_grad.dims(), [1, 1, length]);
    assert_eq!(weight_grad.dims(), [1, 1, 4]);
}

/// Regression test for the asymmetric-padding backward-shape path referenced
/// in https://github.com/tracel-ai/burn/issues/4799. Reduced to a single
/// `conv1d`: the asymmetric path routes through `x.pad(...) -> B::conv1d(padding=0)`,
/// and the backward must restore the original input shape without triggering
/// shape-mismatch failures when the forward drops multiple tail inputs.
#[test]
fn test_conv1d_asymmetric_padding_backward_shape() {
    let device = AutodiffDevice::new();
    let length = 1600;
    let x = TestTensor::<3>::zeros([1, 32, length], &device).require_grad();
    let weight = TestTensor::<3>::zeros([64, 32, 4], &device).require_grad();

    let output = conv1d(
        x.clone(),
        weight.clone(),
        None,
        PaddedConvOptions::asymmetric([4], [3], [0], [1], 1),
    );
    let grads = output.sum().backward();

    assert_eq!(x.grad(&grads).unwrap().dims(), [1, 32, length]);
    assert_eq!(weight.grad(&grads).unwrap().dims(), [64, 32, 4]);
}

struct Conv1dTestCase {
    batch_size: usize,
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    length: usize,
}

struct Grads {
    x: TestTensor<3>,
    weight: TestTensor<3>,
    bias: TestTensor<1>,
}

impl Conv1dTestCase {
    fn assert_grads(self, expected_grads: Grads) {
        let shape_x = Shape::new([self.batch_size, self.channels_in, self.length]);
        let shape_weight = Shape::new([
            self.channels_out,
            self.channels_in / self.groups,
            self.kernel_size,
        ]);
        let device = AutodiffDevice::new();
        let weight = TestTensor::from_data(
            TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                .reshape::<3, _>(shape_weight)
                .into_data(),
            &device,
        )
        .require_grad();
        let bias = TestTensor::from_data(
            TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            &device,
        )
        .require_grad();
        let x = TestTensor::from_data(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                .reshape::<3, _>(shape_x)
                .into_data(),
            &device,
        )
        .require_grad();

        let output = conv1d(
            x.clone(),
            weight.clone(),
            Some(bias.clone()),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups),
        );
        let grads = output.backward();

        // Assert
        let x_grad_actual = x.grad(&grads).unwrap();
        let weight_grad_actual = weight.grad(&grads).unwrap();
        let bias_grad_actual = bias.grad(&grads).unwrap();

        let tolerance = Tolerance::default();
        expected_grads
            .bias
            .to_data()
            .assert_approx_eq::<FloatElem>(&bias_grad_actual.to_data(), tolerance);
        expected_grads
            .weight
            .to_data()
            .assert_approx_eq::<FloatElem>(&weight_grad_actual.to_data(), tolerance);
        expected_grads
            .x
            .to_data()
            .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), tolerance);
    }
}
