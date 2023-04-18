#[burn_tensor_testgen::testgen(module_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv2d;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_conv2d_simple() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            height: 6,
            width: 6,
        };

        test.assert_output(TestTensor::from_floats([
            [
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
            ],
            [
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
            ],
        ]));
    }

    #[test]
    fn test_conv2d_complex() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 4,
            kernel_size_1: 3,
            kernel_size_2: 2,
            padding_1: 1,
            padding_2: 2,
            stride_1: 2,
            stride_2: 3,
            dilation_1: 1,
            dilation_2: 2,
            height: 4,
            width: 5,
        };

        test.assert_output(TestTensor::from_floats([
            [
                [[7., 13., 7.], [10., 19., 10.]],
                [[7., 13., 7.], [10., 19., 10.]],
                [[7., 13., 7.], [10., 19., 10.]],
                [[7., 13., 7.], [10., 19., 10.]],
            ],
            [
                [[7., 13., 7.], [10., 19., 10.]],
                [[7., 13., 7.], [10., 19., 10.]],
                [[7., 13., 7.], [10., 19., 10.]],
                [[7., 13., 7.], [10., 19., 10.]],
            ],
        ]));
    }

    struct Conv2dTestCase {
        batch_size: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        padding_1: usize,
        padding_2: usize,
        stride_1: usize,
        stride_2: usize,
        dilation_1: usize,
        dilation_2: usize,
        height: usize,
        width: usize,
    }

    impl Conv2dTestCase {
        fn assert_output(self, y: TestTensor<4>) {
            let weights = TestTensor::ones([
                self.channels_out,
                self.channels_in,
                self.kernel_size_1,
                self.kernel_size_2,
            ]);
            let bias = TestTensor::ones([self.channels_out]);
            let x = TestTensor::ones([self.batch_size, self.channels_in, self.height, self.width]);
            let output = conv2d(
                x,
                weights,
                Some(bias),
                [self.stride_1, self.stride_2],
                [self.padding_1, self.padding_2],
                [self.dilation_1, self.dilation_2],
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
