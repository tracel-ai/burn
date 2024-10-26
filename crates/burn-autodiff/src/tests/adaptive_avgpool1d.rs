#[burn_tensor_testgen::testgen(ad_adaptive_avg_pool1d)]
mod tests {
    use super::*;
    use burn_tensor::module::adaptive_avg_pool1d;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_avg_pool1d_simple() {
        let test = AdaptiveAvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            length: 5,
            output_size: 3,
        };

        test.assert_output(TestTensor::from_floats(
            [[
                [0.5000, 0.8333, 0.3333, 0.8333, 0.5000],
                [0.5000, 0.8333, 0.3333, 0.8333, 0.5000],
            ]],
            &Default::default(),
        ));
    }

    struct AdaptiveAvgPool1dTestCase {
        batch_size: usize,
        channels: usize,
        length: usize,
        output_size: usize,
    }

    impl AdaptiveAvgPool1dTestCase {
        fn assert_output(self, x_grad: TestTensor<3>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.length]);
            let device = Default::default();
            let x = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<3, _>(shape_x)
                    .into_data(),
                &device,
            )
            .require_grad();
            let output = adaptive_avg_pool1d(x.clone(), self.output_size);
            let grads = output.backward();
            let x_grad_actual = x.grad(&grads).unwrap();

            x_grad
                .to_data()
                .assert_approx_eq(&x_grad_actual.into_data(), 3);
        }
    }
}
