#[burn_tensor_testgen::testgen(module_adaptive_avg_pool1d)]
mod tests {
    use super::*;
    use burn_tensor::module::adaptive_avg_pool1d;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn test_adaptive_avg_pool1d_simple() {
        let test = AdaptiveAvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            length: 8,
            length_out: 4,
        };

        test.assert_output(TestTensor::from_floats([[
            [0.5, 2.5, 4.5, 6.5],
            [8.5, 10.5, 12.5, 14.5],
        ]]));
    }

    #[test]
    fn test_adaptive_avg_pool1d_dyn_filter_size() {
        let test = AdaptiveAvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            length: 7,
            length_out: 3,
        };

        test.assert_output(TestTensor::from_floats([[
            [1.0, 3.0, 5.0],
            [8.0, 10.0, 12.0],
        ]]));
    }

    #[test]
    fn test_adaptive_avg_pool1d_bigger_output() {
        let test = AdaptiveAvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            length: 4,
            length_out: 8,
        };

        test.assert_output(TestTensor::from_floats([[
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
        ]]));
    }

    struct AdaptiveAvgPool1dTestCase {
        batch_size: usize,
        channels: usize,
        length: usize,
        length_out: usize,
    }

    impl AdaptiveAvgPool1dTestCase {
        fn assert_output(self, y: TestTensor<3>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.length]);
            let x = TestTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements())
                    .reshape(shape_x)
                    .into_data()
                    .convert(),
            );
            let output = adaptive_avg_pool1d(x, self.length_out);

            y.into_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
