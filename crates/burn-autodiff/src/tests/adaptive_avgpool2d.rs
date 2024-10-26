#[burn_tensor_testgen::testgen(ad_adaptive_avg_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::module::adaptive_avg_pool2d;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_avg_pool2d_simple() {
        let test = AdaptiveAvgPool2dTestCase {
            batch_size: 1,
            channels: 2,
            height: 5,
            width: 3,
            output_size_1: 3,
            output_size_2: 2,
        };

        test.assert_output(TestTensor::from_floats(
            [[
                [
                    [0.2500, 0.5000, 0.2500],
                    [0.4167, 0.8333, 0.4167],
                    [0.1667, 0.3333, 0.1667],
                    [0.4167, 0.8333, 0.4167],
                    [0.2500, 0.5000, 0.2500],
                ],
                [
                    [0.2500, 0.5000, 0.2500],
                    [0.4167, 0.8333, 0.4167],
                    [0.1667, 0.3333, 0.1667],
                    [0.4167, 0.8333, 0.4167],
                    [0.2500, 0.5000, 0.2500],
                ],
            ]],
            &Default::default(),
        ));
    }

    struct AdaptiveAvgPool2dTestCase {
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        output_size_1: usize,
        output_size_2: usize,
    }

    impl AdaptiveAvgPool2dTestCase {
        fn assert_output(self, x_grad: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
            let device = Default::default();
            let x = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_x)
                    .into_data(),
                &device,
            )
            .require_grad();
            let output = adaptive_avg_pool2d(x.clone(), [self.output_size_1, self.output_size_2]);
            let grads = output.backward();
            let x_grad_actual = x.grad(&grads).unwrap();

            x_grad
                .to_data()
                .assert_approx_eq(&x_grad_actual.into_data(), 3);
        }
    }
}
