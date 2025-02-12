#[burn_tensor_testgen::testgen(module_conv_transpose1d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv_transpose1d;
    use burn_tensor::ops::ConvTransposeOptions;
    use burn_tensor::{Shape, Tensor};

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

        test.assert_output(TestTensor::from_floats(
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

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
