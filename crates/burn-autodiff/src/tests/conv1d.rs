#[burn_tensor_testgen::testgen(ad_conv1d)]
mod tests {
    use super::*;
    use burn_tensor::{module::conv1d, ops::ConvOptions, Shape};

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
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[14., 24., 24., 18.], [26., 42., 42., 30.]],
                    [[14., 24., 24., 18.], [26., 42., 42., 30.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[30., 44., 36.], [54., 76., 60.]],
                    [[30., 44., 36.], [54., 76., 60.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([8., 8.], &device),
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
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[39., 63., 63., 45.], [57., 90., 90., 63.]],
                    [[39., 63., 63., 45.], [57., 90., 90., 63.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[30., 44., 36.], [54., 76., 60.]],
                    [[30., 44., 36.], [54., 76., 60.]],
                    [[30., 44., 36.], [54., 76., 60.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([8., 8., 8.], &device),
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
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[24., 24., 24., 24.], [42., 42., 42., 42.]],
                    [[24., 24., 24., 24.], [42., 42., 42., 42.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[44., 44., 44.], [76., 76., 76.]],
                    [[44., 44., 44.], [76., 76., 76.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([12., 12.], &device),
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
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[8., 16., 8., 10.], [14., 28., 14., 16.]],
                    [[8., 16., 8., 10.], [14., 28., 14., 16.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[10., 20., 24.], [18., 36., 40.]],
                    [[10., 20., 24.], [18., 36., 40.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([4., 4.], &device),
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
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[6., 8., 8., 10.], [12., 14., 14., 16.]],
                    [[6., 8., 8., 10.], [12., 14., 14., 16.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[8., 22., 14.], [16., 38., 22.]],
                    [[8., 22., 14.], [16., 38., 22.]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([4., 4.], &device),
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
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [[1., 3., 3., 3.], [7., 12., 12., 9.]],
                    [[1., 3., 3., 3.], [7., 12., 12., 9.]],
                ],
                &device,
            ),
            weight: TestTensor::from_floats([[[30., 44., 36.]], [[54., 76., 60.]]], &device),
            bias: TestTensor::from_floats([8., 8.], &device),
        };
        test.assert_grads(grads);
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
            let device = Default::default();
            let weight = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape::<3, _>(shape_weight)
                    .into_data(),
                &device,
            )
            .require_grad();
            let bias = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
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

            expected_grads
                .bias
                .to_data()
                .assert_approx_eq(&bias_grad_actual.to_data(), 3);
            expected_grads
                .weight
                .to_data()
                .assert_approx_eq(&weight_grad_actual.to_data(), 3);
            expected_grads
                .x
                .to_data()
                .assert_approx_eq(&x_grad_actual.to_data(), 3);
        }
    }
}
