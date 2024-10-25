#[burn_tensor_testgen::testgen(module_nearest_interpolate)]
mod tests {
    use super::*;
    use burn_tensor::module::interpolate;
    use burn_tensor::ops::{InterpolateMode, InterpolateOptions};
    use burn_tensor::Shape;

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
                [0., 0., 1., 2., 2., 3., 4.],
                [0., 0., 1., 2., 2., 3., 4.],
                [5., 5., 6., 7., 7., 8., 9.],
                [10., 10., 11., 12., 12., 13., 14.],
                [15., 15., 16., 17., 17., 18., 19.],
                [20., 20., 21., 22., 22., 23., 24.],
                [25., 25., 26., 27., 27., 28., 29.],
                [30., 30., 31., 32., 32., 33., 34.],
            ]],
            [[
                [35., 35., 36., 37., 37., 38., 39.],
                [35., 35., 36., 37., 37., 38., 39.],
                [40., 40., 41., 42., 42., 43., 44.],
                [45., 45., 46., 47., 47., 48., 49.],
                [50., 50., 51., 52., 52., 53., 54.],
                [55., 55., 56., 57., 57., 58., 59.],
                [60., 60., 61., 62., 62., 63., 64.],
                [65., 65., 66., 67., 67., 68., 69.],
            ]],
        ]));
    }

    #[test]
    fn test_downsample_interpolation() {
        let test = InterpolateTestCase {
            batch_size: 1,
            channels: 1,
            height: 45,
            width: 14,
            height_out: 4,
            width_out: 6,
        };

        test.assert_output(TestTensor::from([[[
            [0., 2., 4., 7., 9., 11.],
            [154., 156., 158., 161., 163., 165.],
            [308., 310., 312., 315., 317., 319.],
            [462., 464., 466., 469., 471., 473.],
        ]]]));
    }

    #[test]
    fn test_1d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();

        // Run the model
        let input = TestTensor::<3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        let input = input.unsqueeze_dim(2);

        let output = interpolate(
            input,
            [1, 9],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );
        assert_eq!(output.dims(), [1, 1, 1, 9]);

        // assert output data does not contain NaN
        assert!(
            !output
                .clone()
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .any(|&x| x.is_nan()),
            "interpolate output contains NaN"
        );

        TestTensor::<4>::from([[[[
            1.541, 1.541, -0.2934, -2.1788, -2.1788, 0.5684, -1.0845, -1.0845, -1.3986,
        ]]]])
        .to_data()
        .assert_approx_eq(&output.into_data(), 3);
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
        fn assert_output(self, y: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                    .reshape::<4, _>(shape_x)
                    .into_data()
                    .convert::<f32>(),
            );
            let output = interpolate(
                x,
                [self.height_out, self.width_out],
                InterpolateOptions::new(InterpolateMode::Nearest),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
