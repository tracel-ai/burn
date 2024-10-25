#[burn_tensor_testgen::testgen(module_bilinear_interpolate)]
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
                [0.0000, 0.6667, 1.3333, 2.0000, 2.6667, 3.3333, 4.0000],
                [4.2857, 4.9524, 5.6190, 6.2857, 6.9524, 7.6190, 8.2857],
                [8.5714, 9.2381, 9.9048, 10.5714, 11.2381, 11.9048, 12.5714],
                [
                    12.8571, 13.5238, 14.1905, 14.8571, 15.5238, 16.1905, 16.8571,
                ],
                [
                    17.1429, 17.8095, 18.4762, 19.1429, 19.8095, 20.4762, 21.1429,
                ],
                [
                    21.4286, 22.0952, 22.7619, 23.4286, 24.0952, 24.7619, 25.4286,
                ],
                [
                    25.7143, 26.3810, 27.0476, 27.7143, 28.3810, 29.0476, 29.7143,
                ],
                [
                    30.0000, 30.6667, 31.3333, 32.0000, 32.6667, 33.3333, 34.0000,
                ],
            ]],
            [[
                [
                    35.0000, 35.6667, 36.3333, 37.0000, 37.6667, 38.3333, 39.0000,
                ],
                [
                    39.2857, 39.9524, 40.6190, 41.2857, 41.9524, 42.6190, 43.2857,
                ],
                [
                    43.5714, 44.2381, 44.9048, 45.5714, 46.2381, 46.9048, 47.5714,
                ],
                [
                    47.8571, 48.5238, 49.1905, 49.8571, 50.5238, 51.1905, 51.8571,
                ],
                [
                    52.1429, 52.8095, 53.4762, 54.1429, 54.8095, 55.4762, 56.1429,
                ],
                [
                    56.4286, 57.0952, 57.7619, 58.4286, 59.0952, 59.7619, 60.4286,
                ],
                [
                    60.7143, 61.3810, 62.0476, 62.7143, 63.3810, 64.0476, 64.7143,
                ],
                [
                    65.0000, 65.6667, 66.3333, 67.0000, 67.6667, 68.3333, 69.0000,
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
            [0.0, 2.6, 5.2, 7.8, 10.4, 13.],
            [205.3333, 207.9333, 210.5333, 213.1333, 215.7333, 218.3333],
            [410.6667, 413.2667, 415.8667, 418.4667, 421.0667, 423.6667],
            [616., 618.6, 621.2, 623.8, 626.4, 629.],
        ]]]));
    }

    #[test]
    fn test_1d_bilinear() {
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
            InterpolateOptions::new(InterpolateMode::Bilinear),
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
            1.541f32,
            0.39450002,
            -0.76475,
            -1.943125,
            -0.80520004,
            0.36178753,
            -0.671275,
            -1.2022874,
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
                InterpolateOptions::new(InterpolateMode::Bilinear),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
