#[burn_tensor_testgen::testgen(module_bicubic_interpolate)]
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
                [0.0000, 0.5741, 1.3704, 2.0000, 2.6296, 3.4259, 4.0000],
                [4.0015, 4.5755, 5.3718, 6.0015, 6.6311, 7.4274, 8.0015],
                [8.3528, 8.9268, 9.7231, 10.3528, 10.9824, 11.7787, 12.3528],
                [
                    12.7697, 13.3438, 14.1400, 14.7697, 15.3993, 16.1956, 16.7697,
                ],
                [
                    17.2303, 17.8044, 18.6007, 19.2303, 19.8600, 20.6562, 21.2303,
                ],
                [
                    21.6472, 22.2213, 23.0176, 23.6472, 24.2769, 25.0731, 25.6472,
                ],
                [
                    25.9986, 26.5726, 27.3689, 27.9986, 28.6282, 29.4245, 29.9986,
                ],
                [
                    30.0000, 30.5741, 31.3704, 32.0000, 32.6296, 33.4259, 34.0000,
                ],
            ]],
            [[
                [
                    35.0000, 35.5741, 36.3704, 37.0000, 37.6296, 38.4259, 39.0000,
                ],
                [
                    39.0015, 39.5755, 40.3718, 41.0015, 41.6311, 42.4274, 43.0015,
                ],
                [
                    43.3528, 43.9269, 44.7231, 45.3528, 45.9824, 46.7787, 47.3528,
                ],
                [
                    47.7697, 48.3438, 49.1400, 49.7697, 50.3993, 51.1956, 51.7697,
                ],
                [
                    52.2303, 52.8044, 53.6007, 54.2303, 54.8600, 55.6562, 56.2303,
                ],
                [
                    56.6472, 57.2213, 58.0176, 58.6472, 59.2769, 60.0731, 60.6472,
                ],
                [
                    60.9986, 61.5726, 62.3689, 62.9986, 63.6282, 64.4245, 64.9986,
                ],
                [
                    65.0000, 65.5741, 66.3704, 67.0000, 67.6296, 68.4259, 69.0000,
                ],
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
            [0.0000, 2.5760, 5.2480, 7.7520, 10.4240, 13.0000],
            [204.8148, 207.3908, 210.0628, 212.5668, 215.2388, 217.8148],
            [411.1852, 413.7612, 416.4331, 418.9371, 421.6091, 424.1852],
            [616.0000, 618.576, 621.2479, 623.7519, 626.4239, 629.0000],
        ]]]));
    }

    #[test]
    fn test_1d_bicubic() {
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
            InterpolateOptions::new(InterpolateMode::Bicubic),
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
            1.541, 0.5747652, -1.010614, -2.197787, -0.8269969, 0.59609234, -0.5803058, -1.3792794,
            -1.3986,
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
                    .into_data(),
            );
            let output = interpolate(
                x,
                [self.height_out, self.width_out],
                InterpolateOptions::new(InterpolateMode::Bicubic),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
