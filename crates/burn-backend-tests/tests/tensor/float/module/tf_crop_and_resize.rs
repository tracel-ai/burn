use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::interpolate;
use burn_tensor::ops::{CoordinateTransformMode, InterpolateMode, InterpolateOptions};

#[test]
fn test_tf_crop_and_resize_in_bounds() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data(
        [[[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]]],
        &device,
    );

    // ROI: [[0.25, 0.75], [0.25, 0.75]]
    // L_in = 5, L_out = 3
    // y_in = 0.25 * 4 + y_out * (0.5 * 4 / 2) = 1.0 + y_out * 1.0 -> {1.0, 2.0, 3.0}
    // x_in = 0.25 * 4 + x_out * (0.5 * 4 / 2) = 1.0 + x_out * 1.0 -> {1.0, 2.0, 3.0}
    // Output should sample exactly the inner 3x3 block:
    // [[6, 7, 8], [11, 12, 13], [16, 17, 18]]
    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_coordinate_transform(CoordinateTransformMode::TfCropAndResize {
                roi: [[0.25, 0.75], [0.25, 0.75]],
                extrapolation_value: 0.0,
            })
            .with_output_size([3, 3]),
    );

    let expected =
        TestTensor::<4>::from([[[[6.0, 7.0, 8.0], [11.0, 12.0, 13.0], [16.0, 17.0, 18.0]]]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn test_tf_crop_and_resize_extrapolation() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data(
        [[[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]]],
        &device,
    );

    // Height ROI: [-0.5, 1.5]
    // L_in = 5, L_out = 3
    // y_in:
    // y_out = 0 -> -0.5 * 4 = -2.0 (< 0.0: extrapolated)
    // y_out = 1 -> -0.5 * 4 + 1 * (2.0 * 4 / 2) = -2.0 + 4.0 = 2.0 (in bounds)
    // y_out = 2 -> -0.5 * 4 + 2 * (2.0 * 4 / 2) = -2.0 + 8.0 = 6.0 (> 4.0: extrapolated)
    // Width ROI: [0.25, 0.75] -> x_in in {1.0, 2.0, 3.0}
    // Extrapolation value: 99.0
    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_coordinate_transform(CoordinateTransformMode::TfCropAndResize {
                roi: [[-0.5, 1.5], [0.25, 0.75]],
                extrapolation_value: 99.0,
            })
            .with_output_size([3, 3]),
    );

    let expected =
        TestTensor::<4>::from([[[[99.0, 99.0, 99.0], [11.0, 12.0, 13.0], [99.0, 99.0, 99.0]]]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn test_tf_crop_and_resize_output_size_one() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data(
        [[[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]]],
        &device,
    );

    // L_out = 1 formula: 0.5 * (r_start + r_end) * (L_in - 1)
    // With ROI [[0.0, 1.0], [0.0, 1.0]]:
    // y_in = 0.5 * 1.0 * 4 = 2.0
    // x_in = 0.5 * 1.0 * 4 = 2.0
    // Sampled element at (2, 2) is 12.0
    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_coordinate_transform(CoordinateTransformMode::TfCropAndResize {
                roi: [[0.0, 1.0], [0.0, 1.0]],
                extrapolation_value: 0.0,
            })
            .with_output_size([1, 1]),
    );

    let expected = TestTensor::<4>::from([[[[12.0]]]]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn test_tf_crop_and_resize_inverted_roi() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );

    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_coordinate_transform(CoordinateTransformMode::TfCropAndResize {
                roi: [[0.0, 1.0], [1.0, 0.0]],
                extrapolation_value: 99.0,
            })
            .with_output_size([3, 3]),
    );

    let expected = TestTensor::<4>::from([[[[2.0, 1.0, 0.0], [5.0, 4.0, 3.0], [8.0, 7.0, 6.0]]]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn test_tf_crop_and_resize_single_pixel_finite() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data(
        [[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]]],
        &device,
    );

    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_coordinate_transform(CoordinateTransformMode::TfCropAndResize {
                roi: [[0.0, 1.0], [0.0, 1.0]],
                extrapolation_value: 0.0,
            })
            .with_output_size([1, 1]),
    );

    let data = output.into_data().convert::<f32>();
    let val = data.as_slice::<f32>().unwrap()[0];
    assert!(val.is_finite());
    assert!(!val.is_nan());
}
