use super::*;
use burn_tensor::Shape;
use burn_tensor::Tolerance;
use burn_tensor::module::interpolate;
use burn_tensor::ops::{InterpolateMode, InterpolateOptions};

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
            [-0.0000, 0.5685, 1.3918, 2.0000, 2.6082, 3.4315, 4.0000],
            [4.0822, 4.6507, 5.4740, 6.0822, 6.6904, 7.5137, 8.0822],
            [8.7971, 9.3656, 10.1889, 10.7971, 11.4053, 12.2286, 12.7971],
            [
                12.8964, 13.4649, 14.2882, 14.8964, 15.5046, 16.3279, 16.8964,
            ],
            [
                17.1036, 17.6721, 18.4954, 19.1036, 19.7118, 20.5351, 21.1036,
            ],
            [
                21.2029, 21.7715, 22.5947, 23.2029, 23.8112, 24.6344, 25.2029,
            ],
            [
                25.9178, 26.4863, 27.3096, 27.9178, 28.5260, 29.3493, 29.9178,
            ],
            [
                30.0000, 30.5685, 31.3918, 32.0000, 32.6082, 33.4315, 34.0000,
            ],
        ]],
        [[
            [
                35.0000, 35.5685, 36.3918, 37.0000, 37.6082, 38.4315, 39.0000,
            ],
            [
                39.0822, 39.6507, 40.4740, 41.0822, 41.6904, 42.5137, 43.0822,
            ],
            [
                43.7971, 44.3656, 45.1888, 45.7971, 46.4053, 47.2286, 47.7971,
            ],
            [
                47.8964, 48.4649, 49.2882, 49.8964, 50.5046, 51.3279, 51.8964,
            ],
            [
                52.1036, 52.6721, 53.4954, 54.1036, 54.7118, 55.5351, 56.1036,
            ],
            [
                56.2029, 56.7715, 57.5947, 58.2029, 58.8112, 59.6344, 60.2029,
            ],
            [
                60.9178, 61.4863, 62.3096, 62.9178, 63.5260, 64.3493, 64.9178,
            ],
            [
                65.0000, 65.5685, 66.3918, 67.0000, 67.6082, 68.4315, 69.0000,
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
        [-0.0000, 2.6107, 5.1803, 7.8197, 10.3893, 13.0000],
        [205.5606, 208.1713, 210.7408, 213.3802, 215.9498, 218.5606],
        [410.4395, 413.0502, 415.6198, 418.2592, 420.8287, 423.4395],
        [616.0000, 618.6107, 621.1803, 623.8197, 626.3893, 629.0000],
    ]]]));
}

#[test]
fn test_upsample_2x() {
    let test = InterpolateTestCase {
        batch_size: 1,
        channels: 1,
        height: 4,
        width: 4,
        height_out: 8,
        width_out: 8,
    };

    test.assert_output(TestTensor::from([[[
        [
            -0.0000, 0.2972, 0.8164, 1.3131, 1.6869, 2.1836, 2.7028, 3.0000,
        ],
        [
            1.1889, 1.4861, 2.0053, 2.5020, 2.8758, 3.3725, 3.8917, 4.1889,
        ],
        [
            3.2658, 3.5630, 4.0822, 4.5789, 4.9527, 5.4493, 5.9685, 6.2658,
        ],
        [
            5.2524, 5.5496, 6.0689, 6.5655, 6.9393, 7.4360, 7.9552, 8.2524,
        ],
        [
            6.7476, 7.0448, 7.5640, 8.0607, 8.4345, 8.9311, 9.4504, 9.7476,
        ],
        [
            8.7342, 9.0315, 9.5507, 10.0473, 10.4211, 10.9178, 11.4370, 11.7342,
        ],
        [
            10.8111, 11.1083, 11.6275, 12.1242, 12.4980, 12.9947, 13.5139, 13.8111,
        ],
        [
            12.0000, 12.2972, 12.8164, 13.3131, 13.6869, 14.1836, 14.7028, 15.0000,
        ],
    ]]]));
}

#[test]
fn test_upsample_half_pixel() {
    let test = InterpolateTestCase {
        batch_size: 1,
        channels: 1,
        height: 4,
        width: 4,
        height_out: 8,
        width_out: 8,
    };

    test.assert_output_with_align_corners(
        TestTensor::from([[[
            [
                -0.4626, -0.2276, 0.3055, 0.9087, 1.3512, 1.9543, 2.4875, 2.7225,
            ],
            [
                0.4773, 0.7123, 1.2454, 1.8486, 2.2911, 2.8942, 3.4274, 3.6623,
            ],
            [
                2.6099, 2.8449, 3.3780, 3.9812, 4.4237, 5.0268, 5.5600, 5.7949,
            ],
            [
                5.0224, 5.2574, 5.7906, 6.3937, 6.8362, 7.4394, 7.9725, 8.2075,
            ],
            [
                6.7925, 7.0275, 7.5606, 8.1638, 8.6063, 9.2094, 9.7426, 9.9776,
            ],
            [
                9.2051, 9.4400, 9.9732, 10.5763, 11.0188, 11.6220, 12.1551, 12.3901,
            ],
            [
                11.3377, 11.5726, 12.1058, 12.7089, 13.1514, 13.7546, 14.2877, 14.5227,
            ],
            [
                12.2775, 12.5125, 13.0457, 13.6488, 14.0913, 14.6945, 15.2276, 15.4626,
            ],
        ]]]),
        false,
    );
}

#[test]
fn test_1d_lanczos3() {
    let device = Default::default();

    let input = TestTensor::<3>::from_data(
        [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
        &device,
    );

    let input = input.unsqueeze_dim(2);

    let output = interpolate(
        input,
        [1, 9],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    assert_eq!(output.dims(), [1, 1, 1, 9]);

    assert!(
        !output
            .clone()
            .to_data()
            .as_slice::<FloatElem>()
            .unwrap()
            .iter()
            .any(|&x| x.is_nan()),
        "interpolate output contains NaN"
    );

    TestTensor::<4>::from([[[[
        1.5410, 0.7266, -1.1387, -2.2672, -0.7894, 0.6408, -0.4967, -1.4650, -1.3986,
    ]]]])
    .to_data()
    .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::permissive());
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
        self.assert_output_with_align_corners(y, true);
    }

    fn assert_output_with_align_corners(self, y: TestTensor<4>, align_corners: bool) {
        let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
        let x = TestTensor::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                .reshape::<4, _>(shape_x)
                .into_data(),
        );
        let output = interpolate(
            x,
            [self.height_out, self.width_out],
            InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(align_corners),
        );

        let tolerance = Tolerance::permissive();
        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), tolerance);
    }
}
