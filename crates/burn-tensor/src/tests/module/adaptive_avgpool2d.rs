#[burn_tensor_testgen::testgen(module_adaptive_avg_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::module::adaptive_avg_pool2d;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_adaptive_avg_pool2d_simple() {
        let test = AdaptiveAvgPool2dTestCase {
            batch_size: 1,
            channels: 2,
            height: 8,
            width: 6,
            height_out: 4,
            width_out: 4,
        };

        test.assert_output(TestTensor::from([[
            [
                [3.5000, 4.5000, 6.5000, 7.5000],
                [15.5000, 16.5000, 18.5000, 19.5000],
                [27.5000, 28.5000, 30.5000, 31.5000],
                [39.5000, 40.5000, 42.5000, 43.5000],
            ],
            [
                [51.5000, 52.5000, 54.5000, 55.5000],
                [63.5000, 64.5000, 66.5000, 67.5000],
                [75.5000, 76.5000, 78.5000, 79.5000],
                [87.5000, 88.5000, 90.5000, 91.5000],
            ],
        ]]));
    }

    #[test]
    fn test_adaptive_avg_pool2d_dyn_filter_size() {
        let test = AdaptiveAvgPool2dTestCase {
            batch_size: 1,
            channels: 2,
            height: 5,
            width: 7,
            height_out: 3,
            width_out: 2,
        };

        test.assert_output(TestTensor::from([[
            [[5.0000, 8.0000], [15.5000, 18.5000], [26.0000, 29.0000]],
            [[40.0000, 43.0000], [50.5000, 53.5000], [61.0000, 64.0000]],
        ]]));
    }

    #[test]
    fn test_adaptive_avg_pool2d_bigger_output() {
        let test = AdaptiveAvgPool2dTestCase {
            batch_size: 1,
            channels: 2,
            height: 4,
            width: 3,
            height_out: 5,
            width_out: 4,
        };

        test.assert_output(TestTensor::from([[
            [
                [0.0000, 0.5000, 1.5000, 2.0000],
                [1.5000, 2.0000, 3.0000, 3.5000],
                [4.5000, 5.0000, 6.0000, 6.5000],
                [7.5000, 8.0000, 9.0000, 9.5000],
                [9.0000, 9.5000, 10.5000, 11.0000],
            ],
            [
                [12.0000, 12.5000, 13.5000, 14.0000],
                [13.5000, 14.0000, 15.0000, 15.5000],
                [16.5000, 17.0000, 18.0000, 18.5000],
                [19.5000, 20.0000, 21.0000, 21.5000],
                [21.0000, 21.5000, 22.5000, 23.0000],
            ],
        ]]));
    }

    struct AdaptiveAvgPool2dTestCase {
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        height_out: usize,
        width_out: usize,
    }

    impl AdaptiveAvgPool2dTestCase {
        fn assert_output(self, y: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                    .reshape::<4, _>(shape_x)
                    .into_data(),
            );
            let output = adaptive_avg_pool2d(x, [self.height_out, self.width_out]);

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
