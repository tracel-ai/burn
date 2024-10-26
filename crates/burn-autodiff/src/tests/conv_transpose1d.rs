#[burn_tensor_testgen::testgen(ad_conv_transpose1d)]
mod tests {
    use super::*;
    use burn_tensor::{module::conv_transpose1d, ops::ConvTransposeOptions, Shape};

    #[test]
    fn test_conv_transpose1d_basic() {
        let test = ConvTranspose1dTestCase {
            batch_size: 2,
            channels: [2, 2],
            kernel_size: 3,
            padding: 0,
            padding_out: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            size: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[15.0, 15.0, 15.0, 15.0], [51.0, 51.0, 51.0, 51.0]],
                    [[15.0, 15.0, 15.0, 15.0], [51.0, 51.0, 51.0, 51.0]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[44.0, 44.0, 44.0], [44.0, 44.0, 44.0]],
                    [[76.0, 76.0, 76.0], [76.0, 76.0, 76.0]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([12., 12.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv_transpose1d_padding() {
        let test = ConvTranspose1dTestCase {
            batch_size: 2,
            channels: [2, 2],
            kernel_size: 3,
            padding: 2,
            padding_out: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            size: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[7., 12., 8., 3.], [19., 36., 32., 15.]],
                    [[7., 12., 8., 3.], [19., 36., 32., 15.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[26., 22., 18.], [26., 22., 18.]],
                    [[42., 38., 34.], [42., 38., 34.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([4., 4.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv_transpose1d_stride() {
        let test = ConvTranspose1dTestCase {
            batch_size: 2,
            channels: [2, 2],
            kernel_size: 3,
            padding: 0,
            padding_out: 0,
            stride: 2,
            dilation: 1,
            groups: 1,
            size: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[15., 15., 15., 15.], [51., 51., 51., 51.]],
                    [[15., 15., 15., 15.], [51., 51., 51., 51.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[44., 44., 44.], [44., 44., 44.]],
                    [[76., 76., 76.], [76., 76., 76.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([18., 18.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv_transpose1d_stride_padding_out() {
        let test = ConvTranspose1dTestCase {
            batch_size: 2,
            channels: [2, 2],
            kernel_size: 3,
            padding: 0,
            padding_out: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            size: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[15., 15., 15., 15.], [51., 51., 51., 51.]],
                    [[15., 15., 15., 15.], [51., 51., 51., 51.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[44., 44., 44.], [44., 44., 44.]],
                    [[76., 76., 76.], [76., 76., 76.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([20., 20.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv_transpose1d_dilation() {
        let test = ConvTranspose1dTestCase {
            batch_size: 2,
            channels: [2, 2],
            kernel_size: 3,
            padding: 0,
            padding_out: 0,
            stride: 1,
            dilation: 2,
            groups: 1,
            size: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[15., 15., 15., 15.], [51., 51., 51., 51.]],
                    [[15., 15., 15., 15.], [51., 51., 51., 51.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[44., 44., 44.], [44., 44., 44.]],
                    [[76., 76., 76.], [76., 76., 76.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([16., 16.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv_transpose1d_complex() {
        let test = ConvTranspose1dTestCase {
            batch_size: 2,
            channels: [2, 4],
            kernel_size: 3,
            padding: 1,
            padding_out: 1,
            stride: 2,
            dilation: 2,
            groups: 2,
            size: 8,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [
                        [12.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
                        [36.0, 51.0, 51.0, 51.0, 51.0, 51.0, 51.0, 51.0],
                    ],
                    [
                        [12.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
                        [36.0, 51.0, 51.0, 51.0, 51.0, 51.0, 51.0, 51.0],
                    ],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[168.0, 184.0, 184.0], [168.0, 184.0, 184.0]],
                    [[280.0, 312.0, 312.0], [280.0, 312.0, 312.0]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([36.0, 36.0, 36.0, 36.0], &device),
        };
        test.assert_grads(grads);
    }

    struct ConvTranspose1dTestCase {
        batch_size: usize,
        channels: [usize; 2],
        kernel_size: usize,
        padding: usize,
        padding_out: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        size: usize,
    }

    struct Grads {
        x: TestTensor<3>,
        weight: TestTensor<3>,
        bias: TestTensor<1>,
    }

    impl ConvTranspose1dTestCase {
        fn assert_grads(self, expected_grads: Grads) {
            let shape_x = Shape::new([self.batch_size, self.channels[0], self.size]);
            let shape_weight = Shape::new([
                self.channels[0],
                self.channels[1] / self.groups,
                self.kernel_size,
            ]);
            let device = Default::default();
            let weight = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape::<3, _>(shape_weight)
                    .into_data(),
                &device,
            )
            .require_grad();
            let bias = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..self.channels[1] as i64, &device).into_data(),
                &device,
            )
            .require_grad();
            let x = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<3, _>(shape_x)
                    .into_data(),
                &device,
            )
            .require_grad();
            let output = conv_transpose1d(
                x.clone(),
                weight.clone(),
                Some(bias.clone()),
                ConvTransposeOptions::new(
                    [self.stride],
                    [self.padding],
                    [self.padding_out],
                    [self.dilation],
                    self.groups,
                ),
            );
            let grads = output.backward();

            // Assert
            let x_grad_actual = x.grad(&grads).unwrap();
            let weight_grad_actual = weight.grad(&grads).unwrap();
            let bias_grad_actual = bias.grad(&grads).unwrap();

            expected_grads
                .bias
                .to_data()
                .assert_approx_eq(&bias_grad_actual.to_data(), 3);
            expected_grads
                .x
                .to_data()
                .assert_approx_eq(&x_grad_actual.to_data(), 3);
            expected_grads
                .weight
                .to_data()
                .assert_approx_eq(&weight_grad_actual.to_data(), 3);
        }
    }
}
