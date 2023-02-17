#[burn_tensor_testgen::testgen(ad_conv1d)]
mod tests {
    use super::*;
    use burn_tensor::{module::conv1d, Data};

    #[test]
    fn test_conv1d_basic() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 3,
            kernel_size: 3,
            padding: 1,
            stride: 1,
            length: 6,
        };
        let grads = Grads {
            x: TestTensor::from_floats([
                [
                    [6., 9., 9., 9., 9., 6.],
                    [6., 9., 9., 9., 9., 6.],
                    [6., 9., 9., 9., 9., 6.],
                ],
                [
                    [6., 9., 9., 9., 9., 6.],
                    [6., 9., 9., 9., 9., 6.],
                    [6., 9., 9., 9., 9., 6.],
                ],
            ]),
            weight: TestTensor::from_floats([
                [[10., 12., 10.], [10., 12., 10.], [10., 12., 10.]],
                [[10., 12., 10.], [10., 12., 10.], [10., 12., 10.]],
                [[10., 12., 10.], [10., 12., 10.], [10., 12., 10.]],
            ]),
            bias: TestTensor::from_floats([12., 12., 12.]),
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
            length: 6,
        };
        let grads = Grads {
            x: TestTensor::from_floats([
                [[6., 9., 9., 9., 9., 6.], [6., 9., 9., 9., 9., 6.]],
                [[6., 9., 9., 9., 9., 6.], [6., 9., 9., 9., 9., 6.]],
            ]),
            weight: TestTensor::from_floats([
                [[10., 12., 10.], [10., 12., 10.]],
                [[10., 12., 10.], [10., 12., 10.]],
                [[10., 12., 10.], [10., 12., 10.]],
            ]),
            bias: TestTensor::from_floats([12., 12., 12.]),
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
            length: 6,
        };
        let grads = Grads {
            x: TestTensor::from_floats([
                [[6., 6., 6., 6., 6., 6.], [6., 6., 6., 6., 6., 6.]],
                [[6., 6., 6., 6., 6., 6.], [6., 6., 6., 6., 6., 6.]],
            ]),
            weight: TestTensor::from_floats([
                [[12., 12., 12.], [12., 12., 12.]],
                [[12., 12., 12.], [12., 12., 12.]],
            ]),
            bias: TestTensor::from_floats([16., 16.]),
        };
        test.assert_grads(grads);
    }

    #[ignore = "Stride different than 1 is not supported yet."]
    #[test]
    fn test_conv1d_with_stride() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 2,
            kernel_size: 3,
            padding: 1,
            stride: 2,
            length: 8,
        };
        let grads = Grads {
            x: TestTensor::from_floats([
                [[2., 4., 2., 4., 2., 2.], [2., 4., 2., 4., 2., 2.]],
                [[2., 4., 2., 4., 2., 2.], [2., 4., 2., 4., 2., 2.]],
            ]),
            weight: TestTensor::from_floats([
                [[4., 6., 6.], [4., 6., 6.]],
                [[4., 6., 6.], [4., 6., 6.]],
            ]),
            bias: TestTensor::from_floats([6., 6.]),
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
        length: usize,
    }

    struct Grads {
        x: TestTensor<3>,
        weight: TestTensor<3>,
        bias: TestTensor<1>,
    }

    impl Conv1dTestCase {
        fn assert_grads(self, expected_grads: Grads) {
            let weight =
                TestADTensor::ones([self.channels_out, self.channels_in, self.kernel_size]);
            let bias = TestADTensor::ones([self.channels_out]);
            let x = TestADTensor::ones([self.batch_size, self.channels_in, self.length]);
            let output = conv1d(
                x.clone(),
                weight.clone(),
                Some(bias.clone()),
                self.stride,
                self.padding,
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
