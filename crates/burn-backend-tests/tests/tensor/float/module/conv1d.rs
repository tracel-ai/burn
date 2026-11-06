use super::*;
use burn_backend::ops::ModuleOps;
use burn_dispatch::Dispatch;
use burn_tensor::Shape;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::conv1d;
use burn_tensor::ops::ConvOptions;

#[test]
fn test_conv1d_bias_backward_channels_last() {
    conv1d_bias_backward(channels_last_bias_output_grad())
        .into_data()
        .assert_eq(&TensorData::from([14.0, 16.0, 22.0]), false);
}

#[test]
fn test_conv1d_bias_backward_bf16() {
    let grad = channels_last_bias_output_grad();
    if !grad.device().supports_dtype(burn_tensor::DType::BF16) {
        return;
    }
    conv1d_bias_backward(grad.cast(burn_tensor::DType::BF16))
        .into_data()
        .assert_eq(&TensorData::from([14.0, 16.0, 22.0]), false);
}

#[test]
fn test_conv1d_bias_backward_cancels_large_spatial_sums() {
    // Each spatial sum in channel 0 overflows f16, but the complete sum is zero.
    let grad = TestTensor::from_data(
        [
            [[32768.0; 8], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
            [
                [-32768.0; 8],
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 8.0],
            ],
        ],
        &Default::default(),
    );

    conv1d_bias_backward(grad)
        .into_data()
        .assert_eq(&TensorData::from([0.0, 16.0]), false);
}

fn channels_last_bias_output_grad() -> TestTensor<3> {
    // Store NLC, then present NCL.
    TestTensor::from_data(
        [
            [
                [1.0, -5.0, 9.0],
                [2.0, 6.0, -10.0],
                [-3.0, 7.0, 11.0],
                [4.0, -8.0, 12.0],
            ],
            [
                [-2.0, 6.0, -10.0],
                [3.0, -7.0, 11.0],
                [4.0, 8.0, 12.0],
                [5.0, 9.0, -13.0],
            ],
        ],
        &Default::default(),
    )
    .permute([0, 2, 1])
}

// Reduces [batch, channels, length] to [channels], preserving dtype and device.
fn conv1d_bias_backward(output_grad: TestTensor<3>) -> TestTensor<1> {
    let device = output_grad.device();
    let dtype = output_grad.dtype();
    let [batch, channels, length] = output_grad.dims();
    let input = TestTensor::<3>::zeros([batch, 1, length], (&device, dtype));
    let bias = TestTensor::<1>::zeros([channels], (&device, dtype));
    let actual = TestTensor::from_dispatch(Dispatch::conv1d_bias_backward(
        input.into_dispatch(),
        bias.into_dispatch(),
        output_grad.into_dispatch(),
    ));
    assert_eq!(actual.dtype(), dtype);
    assert_eq!(actual.device(), device);
    actual
}

#[test]
fn test_conv1d_simple() {
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

    test.assert_output(TestTensor::from([
        [[43., 67., 82., 49.], [104., 176., 227., 158.]],
        [[139., 187., 202., 113.], [392., 584., 635., 414.]],
    ]));
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

    test.assert_output(TestTensor::from([
        [[62., 38.], [159., 111.]],
        [[158., 102.], [447., 367.]],
    ]));
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

    test.assert_output(TestTensor::from([
        [[2., 5., 8., 3.], [42., 63., 75., 47.]],
        [[26., 29., 32., 11.], [114., 159., 171., 103.]],
    ]));
}

#[test]
fn test_conv1d_complex() {
    let test = Conv1dTestCase {
        batch_size: 2,
        channels_in: 3,
        channels_out: 4,
        kernel_size: 3,
        padding: 1,
        stride: 2,
        dilation: 1,
        groups: 1,
        length: 4,
    };

    test.assert_output(TestTensor::from_data(
        [
            [[171., 294.], [415., 781.], [659., 1268.], [903., 1755.]],
            [[495., 726.], [1387., 2185.], [2279., 3644.], [3171., 5103.]],
        ],
        &Default::default(),
    ));
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

impl Conv1dTestCase {
    fn assert_output(self, y: TestTensor<3>) {
        let shape_x = Shape::new([self.batch_size, self.channels_in, self.length]);
        let shape_weight = Shape::new([
            self.channels_out,
            self.channels_in / self.groups,
            self.kernel_size,
        ]);
        let device = Default::default();
        let weight = TestTensor::from_data(
            TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                .reshape::<3, _>(shape_weight)
                .into_data(),
            &device,
        );
        let bias = TestTensor::from_data(
            TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            &device,
        );
        let x = TestTensor::from_data(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                .reshape::<3, _>(shape_x)
                .into_data(),
            &device,
        );
        let output = conv1d(
            x,
            weight,
            Some(bias),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups),
        );

        let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(1e-3);
        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), tolerance);
    }
}
