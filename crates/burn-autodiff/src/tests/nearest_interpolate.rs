#[burn_tensor_testgen::testgen(ad_nearest_interpolate)]
mod tests {
    use super::*;
    use burn_tensor::module::interpolate;
    use burn_tensor::ops::{InterpolateMode, InterpolateOptions};
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_upsample_interpolation() {
        let test = InterpolateTestCase {
            batch_size: 2,
            channels: 1,
            height: 7,
            width: 5,
            height_out: 8,
            width_out: 7,
        };

        test.assert_output(TestTensor::from([
            [[
                [4., 2., 4., 2., 2.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
            ]],
            [[
                [4., 2., 4., 2., 2.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
                [2., 1., 2., 1., 1.],
            ]],
        ]));
    }

    #[test]
    fn test_downsample_interpolation() {
        let test = InterpolateTestCase {
            batch_size: 1,
            channels: 1,
            height: 8,
            width: 8,
            height_out: 4,
            width_out: 6,
        };

        test.assert_output(TestTensor::from([[[
            [1., 1., 1., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
        ]]]));
    }

    struct InterpolateTestCase {
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        height_out: usize,
        width_out: usize,
    }

    impl InterpolateTestCase {
        fn assert_output(self, x_grad: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
            let device = Default::default();
            let x = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &x_grad.device())
                    .reshape::<4, _>(shape_x)
                    .into_data(),
                &device,
            )
            .require_grad();

            let output = interpolate(
                x.clone(),
                [self.height_out, self.width_out],
                InterpolateOptions::new(InterpolateMode::Nearest),
            );

            let grads = output.backward();
            let x_grad_actual = x.grad(&grads).unwrap();

            x_grad
                .to_data()
                .assert_approx_eq(&x_grad_actual.into_data(), 3);
        }
    }
}
