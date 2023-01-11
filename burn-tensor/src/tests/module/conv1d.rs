#[burn_tensor_testgen::testgen(module_conv1d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv1d;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_conv1d_simple() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 3,
            kernel_size: 3,
            padding: 1,
            stride: 1,
            length: 6,
        };

        test.assert_output(TestTensor::from_floats([
            [
                [7., 10., 10., 10., 10., 7.],
                [7., 10., 10., 10., 10., 7.],
                [7., 10., 10., 10., 10., 7.],
            ],
            [
                [7., 10., 10., 10., 10., 7.],
                [7., 10., 10., 10., 10., 7.],
                [7., 10., 10., 10., 10., 7.],
            ],
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
            length: 9,
        };

        test.assert_output(TestTensor::from_floats([
            [
                [7., 10., 10., 10., 7.],
                [7., 10., 10., 10., 7.],
                [7., 10., 10., 10., 7.],
                [7., 10., 10., 10., 7.],
            ],
            [
                [7., 10., 10., 10., 7.],
                [7., 10., 10., 10., 7.],
                [7., 10., 10., 10., 7.],
                [7., 10., 10., 10., 7.],
            ],
        ]));
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

    impl Conv1dTestCase {
        fn assert_output(self, y: TestTensor<3>) {
            let weights = TestTensor::ones([self.channels_out, self.channels_in, self.kernel_size]);
            let bias = TestTensor::ones([self.channels_out]);
            let x = TestTensor::ones([self.batch_size, self.channels_in, self.length]);
            let output = conv1d(&x, &weights, Some(&bias), self.stride, self.padding);

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
