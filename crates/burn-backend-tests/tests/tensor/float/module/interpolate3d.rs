use super::*;
use burn_tensor::module::interpolate3d;
use burn_tensor::ops::{CoordinateTransformMode, Interpolate3dMode, Interpolate3dOptions};
use burn_tensor::{DType, TensorData, Tolerance};

// References use torch.nn.functional.interpolate with float64 inputs and the same sizes/scales.

#[test]
fn trilinear_half_pixel_batched_channels_permuted() {
    let device = Default::default();
    // torch.tensor([(i * 7 % 19) - 9 for i in range(48)]).reshape(2, 2, 2, 3, 2)
    //     .permute(0, 1, 2, 4, 3), size=(3, 1, 2), mode="trilinear", align_corners=False
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            (0..48).map(|i| (i * 7 % 19) as f32 - 9.0).collect(),
            [2, 2, 2, 3, 2],
        ),
        &device,
    )
    .permute([0, 1, 2, 4, 3]);
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::HalfPixel,
        })
        .with_output_size([3, 1, 2]),
    );
    let expected = TensorData::new(
        vec![
            -4.375f32, 2.375, -2.375, 0.8125, -0.375, -0.75, 1.25, 0.875, -0.3125, -0.6875, -1.875,
            -2.25, 2.125, 1.75, -0.625, 2.5625, -3.375, 3.375, -1.75, -2.125, 0.25, -0.125, 2.25,
            1.875,
        ],
        [2, 2, 3, 1, 2],
    );
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn trilinear_align_corners_singleton_axes() {
    let device = Default::default();
    let input = TestTensor::<5>::from_data(
        TensorData::new(vec![2.0f32, -3.0, 7.0, 5.0, 11.0, -1.0], [1, 1, 1, 2, 3]),
        &device,
    );
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::AlignCorners,
        })
        .with_output_size([2, 1, 4]),
    );
    let expected = TensorData::new(
        vec![
            2.0f32,
            -4.0 / 3.0,
            1.0 / 3.0,
            7.0,
            2.0,
            -4.0 / 3.0,
            1.0 / 3.0,
            7.0,
        ],
        [1, 1, 2, 1, 4],
    );
    // Half-precision coordinate rounding is amplified by cancellation in this fixture.
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_absolute(5e-3),
    );
}

fn assert_nearest(mode: Interpolate3dMode, expected: [f32; 6]) {
    let device = Default::default();
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            vec![2.0f32, -3.0, 7.0, 1.0, -5.0, 9.0, 4.0, 11.0],
            [1, 1, 4, 1, 2],
        ),
        &device,
    );
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(mode).with_output_size([2, 1, 3]),
    );
    output.into_data().assert_eq(
        &TensorData::new(expected.to_vec(), [1, 1, 2, 1, 3]).convert::<FloatElem>(),
        false,
    );
}

#[test]
fn nearest_uses_asymmetric_floor() {
    assert_nearest(
        Interpolate3dMode::Nearest,
        [2.0, 2.0, -3.0, -5.0, -5.0, 9.0],
    );
}

#[test]
fn nearest_exact_uses_half_pixel_ties() {
    assert_nearest(
        Interpolate3dMode::NearestExact,
        [7.0, 1.0, 1.0, 4.0, 11.0, 11.0],
    );
}

#[test]
fn trilinear_scale_factors() {
    let device = Default::default();
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            vec![
                2.0f32, -3.0, 7.0, 5.0, 11.0, -1.0, -4.0, 8.0, 3.0, 6.0, -7.0, 9.0,
            ],
            [1, 1, 2, 2, 3],
        ),
        &device,
    );
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::HalfPixel,
        })
        .with_scale_factor([1.5, 0.5, 2.0]),
    );
    let expected = TensorData::new(
        vec![
            3.5f32, 3.625, 3.875, 3.75, 3.25, 3.0, 2.25, 2.25, 2.25, 2.8125, 3.9375, 4.5, 1.0,
            0.875, 0.625, 1.875, 4.625, 6.0,
        ],
        [1, 1, 3, 1, 6],
    );
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[cfg(feature = "flex")]
fn trilinear_preserves_fractional_scales() {
    let device = burn_tensor::Device::flex();
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            (0..18).map(|i| (i * 7 % 19) as f32 - 9.0).collect(),
            [1, 1, 3, 3, 2],
        ),
        &device,
    )
    .cast(<FloatElem as burn_tensor::Element>::dtype());
    // recompute_scale_factor=False: use the supplied scales, not rounded output/input ratios.
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::HalfPixel,
        })
        .with_scale_factor([1.75, 0.75, 1.6]),
    );
    let expected = TensorData::new(
        vec![
            -6.666_666_5f32,
            -4.989_583_5,
            -2.833_333_3,
            2.5,
            1.40625,
            0.0,
            -5.238_095_3,
            -3.561_012,
            -1.404_761_9,
            3.928_571_5,
            1.350_446_5,
            -1.964_285_7,
            -2.952_381,
            -1.275_297_6,
            0.880_952_36,
            6.214_286,
            1.261_160_7,
            -5.107_143,
            -2.25,
            0.119_791_664,
            3.166_666_7,
            3.75,
            0.578_125,
            -3.5,
            -1.833_333_4,
            1.229_166_6,
            5.166_666_5,
            1.0,
            -0.09375,
            -1.5,
        ],
        [1, 1, 5, 2, 3],
    );
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn trilinear_does_not_mix_neighboring_nan_columns() {
    let device = Default::default();
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            vec![
                2.0f32,
                4.0,
                f32::NAN,
                f32::NAN,
                6.0,
                8.0,
                f32::NAN,
                f32::NAN,
            ],
            [1, 1, 2, 1, 4],
        ),
        &device,
    );
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::HalfPixel,
        })
        .with_output_size([3, 1, 2]),
    );
    // PyTorch: [3., NaN, 5., NaN, 7., NaN]. The two output columns have disjoint input taps.
    let values = output.into_data().try_into_vec_as::<f32>().unwrap();
    assert_eq!([values[0], values[2], values[4]], [3.0, 5.0, 7.0]);
    assert!(values[1].is_nan() && values[3].is_nan() && values[5].is_nan());
}

#[test]
fn interpolate3d_empty_batch() {
    let input = TestTensor::<5>::empty([0, 2, 2, 3, 4], &Default::default());
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Nearest).with_output_size([3, 1, 2]),
    );
    output.into_data().assert_eq(
        &TensorData::new(Vec::<FloatElem>::new(), [0, 2, 3, 1, 2]),
        false,
    );
}

#[test]
fn interpolate3d_empty_channels() {
    let input = TestTensor::<5>::empty([2, 0, 2, 3, 4], &Default::default());
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::HalfPixel,
        })
        .with_scale_factor([1.5, 2.0, 0.5]),
    );
    output.into_data().assert_eq(
        &TensorData::new(Vec::<FloatElem>::new(), [2, 0, 3, 6, 2]),
        false,
    );
}

#[test]
fn interpolate3d_preserves_runtime_dtype() {
    let input = TestTensor::<5>::from_data(
        TensorData::new(vec![0.5f32, -1.25, 2.5, 3.75], [1, 1, 2, 1, 2]),
        &Default::default(),
    )
    .cast(DType::F32); // Differs from the device default in tensor_f16.
    let output = interpolate3d(
        input,
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform: CoordinateTransformMode::HalfPixel,
        })
        .with_output_size([3, 1, 2]),
    );
    assert_eq!(output.dtype(), DType::F32);
    output.into_data().assert_eq(
        &TensorData::new(vec![0.5f32, -1.25, 1.5, 1.25, 2.5, 3.75], [1, 1, 3, 1, 2]),
        false,
    );
}
