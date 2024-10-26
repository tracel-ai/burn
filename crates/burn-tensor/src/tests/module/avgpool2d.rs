#[burn_tensor_testgen::testgen(module_avg_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::module::avg_pool2d;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_avg_pool2d_simple() {
        let test = AvgPool2dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            height: 6,
            width: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from([[[
            [7., 8., 9., 10.],
            [13., 14., 15., 16.],
            [19., 20., 21., 22.],
            [25., 26., 27., 28.],
        ]]]));
    }

    #[test]
    fn test_avg_pool2d_complex() {
        let test = AvgPool2dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size_1: 3,
            kernel_size_2: 4,
            padding_1: 1,
            padding_2: 2,
            stride_1: 1,
            stride_2: 2,
            height: 4,
            width: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from([[[
            [1.1667, 3.0000, 4.3333, 2.5000],
            [3.2500, 7.5000, 9.5000, 5.2500],
            [6.2500, 13.5000, 15.5000, 8.2500],
            [5.1667, 11.0000, 12.3333, 6.5000],
        ]]]));
    }

    #[test]
    fn test_avg_pool2d_complex_dont_include_pad() {
        let test = AvgPool2dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size_1: 3,
            kernel_size_2: 4,
            padding_1: 1,
            padding_2: 2,
            stride_1: 1,
            stride_2: 2,
            height: 4,
            width: 6,
            count_include_pad: false,
        };

        test.assert_output(TestTensor::from([[[
            [3.5000, 4.5000, 6.5000, 7.5000],
            [6.5000, 7.5000, 9.5000, 10.5000],
            [12.5000, 13.5000, 15.5000, 16.5000],
            [15.5000, 16.5000, 18.5000, 19.5000],
        ]]]));
    }

    struct AvgPool2dTestCase {
        batch_size: usize,
        channels: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        padding_1: usize,
        padding_2: usize,
        stride_1: usize,
        stride_2: usize,
        height: usize,
        width: usize,
        count_include_pad: bool,
    }

    impl AvgPool2dTestCase {
        fn assert_output(self, y: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                    .reshape::<4, _>(shape_x)
                    .into_data(),
            );
            let output = avg_pool2d(
                x,
                [self.kernel_size_1, self.kernel_size_2],
                [self.stride_1, self.stride_2],
                [self.padding_1, self.padding_2],
                self.count_include_pad,
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
