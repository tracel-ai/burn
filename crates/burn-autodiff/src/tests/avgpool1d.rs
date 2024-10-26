#[burn_tensor_testgen::testgen(ad_avg_pool1d)]
mod tests {
    use super::*;
    use burn_tensor::module::avg_pool1d;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_avg_pool1d_simple() {
        let test = AvgPool1dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size: 3,
            padding: 0,
            stride: 1,
            length: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from_floats(
            [[[0.3333, 0.6667, 1.0000, 1.0000, 0.6667, 0.3333]]],
            &Default::default(),
        ));
    }

    #[test]
    fn test_avg_pool1d_complex() {
        let test = AvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            kernel_size: 3,
            padding: 1,
            stride: 2,
            length: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from_floats(
            [[
                [0.3333, 0.6667, 0.3333, 0.6667, 0.3333, 0.3333],
                [0.3333, 0.6667, 0.3333, 0.6667, 0.3333, 0.3333],
            ]],
            &Default::default(),
        ));
    }

    #[test]
    fn test_avg_pool1d_complex_dont_count_pad() {
        let test = AvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            kernel_size: 3,
            padding: 1,
            stride: 2,
            length: 6,
            count_include_pad: false,
        };

        test.assert_output(TestTensor::from_floats(
            [[
                [0.5000, 0.8333, 0.3333, 0.6667, 0.3333, 0.3333],
                [0.5000, 0.8333, 0.3333, 0.6667, 0.3333, 0.3333],
            ]],
            &Default::default(),
        ));
    }

    struct AvgPool1dTestCase {
        batch_size: usize,
        channels: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        length: usize,
        count_include_pad: bool,
    }

    impl AvgPool1dTestCase {
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
            let output = avg_pool1d(
                x.clone(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.count_include_pad,
            );
            let grads = output.backward();
            let x_grad_actual = x.grad(&grads).unwrap();

            x_grad
                .to_data()
                .assert_approx_eq(&x_grad_actual.into_data(), 3);
        }
    }
}
