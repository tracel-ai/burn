use super::*;
use burn_backend::ops::ModuleOps;
use burn_dispatch::Dispatch;
use burn_tensor::Shape;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::conv_transpose1d;
use burn_tensor::ops::ConvTransposeOptions;

#[test]
fn test_conv_transpose1d_bias_backward_channels_last() {
    conv_transpose1d_bias_backward(channels_last_bias_output_grad())
        .into_data()
        .assert_eq(&TensorData::from([14.0, 16.0, 22.0]), false);
}

#[test]
fn test_conv_transpose1d_bias_backward_bf16() {
    let grad = channels_last_bias_output_grad();
    if !grad.device().supports_dtype(burn_tensor::DType::BF16) {
        return;
    }
    conv_transpose1d_bias_backward(grad.cast(burn_tensor::DType::BF16))
        .into_data()
        .assert_eq(&TensorData::from([14.0, 16.0, 22.0]), false);
}

#[test]
fn test_conv_transpose1d_bias_backward_cancels_large_spatial_sums() {
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

    conv_transpose1d_bias_backward(grad)
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
fn conv_transpose1d_bias_backward(output_grad: TestTensor<3>) -> TestTensor<1> {
    let device = output_grad.device();
    let dtype = output_grad.dtype();
    let [batch, channels, length] = output_grad.dims();
    let input = TestTensor::<3>::zeros([batch, 1, length], (&device, dtype));
    let bias = TestTensor::<1>::zeros([channels], (&device, dtype));
    let actual = TestTensor::from_dispatch(Dispatch::conv_transpose1d_bias_backward(
        input.into_dispatch(),
        bias.into_dispatch(),
        output_grad.into_dispatch(),
    ));
    assert_eq!(actual.dtype(), dtype);
    assert_eq!(actual.device(), device);
    actual
}

#[test]
fn test_conv_transpose1d_diff_channels() {
    let test = ConvTranspose1dTestCase {
        batch_size: 1,
        channels_in: 3,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        padding_out: 0,
        stride: 1,
        dilation: 1,
        groups: 1,
        length: 4,
    };

    test.assert_output(TestTensor::from([[
        [270., 453., 516., 387.],
        [352., 589., 679., 505.],
    ]]));
}

#[test]
fn test_conv_transpose1d_stride() {
    let test = ConvTranspose1dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        padding_out: 1,
        stride: 2,
        dilation: 1,
        groups: 1,
        length: 4,
    };

    test.assert_output(TestTensor::from([[
        [28., 62., 36., 78., 44., 94., 52., 62.],
        [41., 93., 55., 121., 69., 149., 83., 93.],
    ]]));
}

#[test]
fn test_conv_transpose1d_dilation() {
    let test = ConvTranspose1dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        padding_out: 0,
        stride: 1,
        dilation: 2,
        groups: 1,
        length: 4,
    };

    test.assert_output(TestTensor::from([[
        [30., 64., 78., 76., 94., 52.],
        [49., 101., 127., 113., 143., 77.],
    ]]));
}

#[test]
fn test_conv_transpose1d_groups() {
    let test = ConvTranspose1dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 2,
        kernel_size: 3,
        padding: 1,
        padding_out: 0,
        stride: 1,
        dilation: 1,
        groups: 2,
        length: 4,
    };

    test.assert_output(TestTensor::from_data(
        [[[0., 1., 4., 7.], [32., 59., 71., 59.]]],
        &Default::default(),
    ));
}

struct ConvTranspose1dTestCase {
    batch_size: usize,
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    padding: usize,
    padding_out: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    length: usize,
}

impl ConvTranspose1dTestCase {
    fn assert_output(self, y: TestTensor<3>) {
        let shape_x = Shape::new([self.batch_size, self.channels_in, self.length]);
        let shape_weights = Shape::new([
            self.channels_in,
            self.channels_out / self.groups,
            self.kernel_size,
        ]);
        let device = Default::default();
        let weights = TestTensor::from_data(
            TestTensorInt::arange(0..shape_weights.num_elements() as i64, &device)
                .reshape::<3, _>(shape_weights)
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
        let output = conv_transpose1d(
            x,
            weights,
            Some(bias),
            ConvTransposeOptions::new(
                [self.stride],
                [self.padding],
                [self.padding_out],
                [self.dilation],
                self.groups,
            ),
        );

        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    }
}
