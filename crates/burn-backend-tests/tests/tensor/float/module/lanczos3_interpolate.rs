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
            [-0.0000, 0.6119, 1.3424, 2.0000, 2.6400, 3.3705, 4.0000],
            [4.2127, 4.8054, 5.5350, 6.2103, 6.8310, 7.5607, 8.2078],
            [8.6814, 9.2529, 9.9808, 10.6741, 11.2737, 12.0015, 12.6668],
            [
                12.8262, 13.3784, 14.1049, 14.8153, 15.3955, 16.1220, 16.8044,
            ],
            [
                17.0105, 17.5443, 18.2708, 18.9996, 19.5614, 20.2879, 20.9887,
            ],
            [
                21.2093, 21.7258, 22.4537, 23.2020, 23.7466, 24.4744, 25.1947,
            ],
            [
                25.7508, 26.2489, 26.9785, 27.7484, 28.2745, 29.0042, 29.7460,
            ],
            [
                30.0000, 30.4801, 31.2106, 32.0000, 32.5083, 33.2388, 34.0000,
            ],
        ]],
        [[
            [
                35.0000, 35.4582, 36.1887, 37.0000, 37.4863, 38.2168, 39.0000,
            ],
            [
                39.1702, 39.6093, 40.3389, 41.1677, 41.6349, 42.3646, 43.1653,
            ],
            [
                43.5539, 43.9723, 44.7001, 45.5466, 45.9930, 46.7209, 47.5393,
            ],
            [
                47.6356, 48.0349, 48.7614, 49.6247, 50.0520, 50.7785, 51.6138,
            ],
            [
                51.8199, 52.2008, 52.9273, 53.8090, 54.2179, 54.9444, 55.7981,
            ],
            [
                56.0818, 56.4452, 57.1730, 58.0745, 58.4659, 59.1938, 60.0672,
            ],
            [
                60.7083, 61.0528, 61.7824, 62.7058, 63.0785, 63.8081, 64.7034,
            ],
            [
                65.0000, 65.3264, 66.0569, 67.0000, 67.3545, 68.0851, 69.0000,
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
        [-0.0000, 2.5971, 5.1692, 7.8029, 10.3352, 13.0000],
        [204.6577, 206.1778, 209.3642, 211.9864, 213.8819, 217.6006],
        [408.6368, 409.0948, 412.9048, 415.5269, 416.7989, 421.5797],
        [616.0000, 615.3897, 619.8449, 622.4787, 623.1278, 629.0000],
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
                -0.3630, -0.1342, 0.4218, 0.9549, 1.4461, 1.9792, 2.5352, 2.7640,
            ],
            [
                0.5522, 0.7810, 1.3370, 1.8702, 2.3613, 2.8944, 3.4505, 3.6793,
            ],
            [
                2.7763, 3.0051, 3.5611, 4.0943, 4.5854, 5.1185, 5.6746, 5.9034,
            ],
            [
                4.9088, 5.1376, 5.6937, 6.2268, 6.7179, 7.2511, 7.8071, 8.0359,
            ],
            [
                6.8734, 7.1022, 7.6582, 8.1914, 8.6825, 9.2156, 9.7717, 10.0005,
            ],
            [
                9.0059, 9.2347, 9.7908, 10.3239, 10.8150, 11.3482, 11.9042, 12.1330,
            ],
            [
                11.2300, 11.4588, 12.0148, 12.5480, 13.0391, 13.5723, 14.1283, 14.3571,
            ],
            [
                12.1453, 12.3741, 12.9301, 13.4632, 13.9544, 14.4875, 15.0435, 15.2723,
            ],
        ]]]),
        false,
    );
}

#[test]
fn test_1d_lanczos3() {
    let device = Default::default();

    let input = TestTensor::<3>::from_floats(
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
        1.5410, 0.6498, -1.0548, -2.2672, -0.7894, 0.6408, -0.5223, -1.4650, -1.3986,
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
