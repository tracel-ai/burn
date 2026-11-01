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
    let input = TestTensor::<3>::from_data(
        [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
        &device,
    );

    let input = input.unsqueeze_dim(2);

    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([1, 9]),
    );
    assert_eq!(output.dims(), [1, 1, 1, 9]);

    // assert output data does not contain NaN
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
        1.541, 1.541, -0.2934, -2.1788, -2.1788, 0.5684, -1.0845, -1.0845, -1.3986,
    ]]]])
    .to_data()
    .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
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
            InterpolateOptions::new(InterpolateMode::Nearest)
                .with_output_size([self.height_out, self.width_out]),
        );

        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    }
}

#[test]
fn test_nearest_downsample_f64_precision_coordinate_mapping() {
    let device = Default::default();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    let input = TestTensorInt::arange(0..14, &device)
        .float()
        .reshape([1, 1, 1, 14]);

    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([1, 2]),
    );

    // In f32, 1.0 / (2.0 / 14.0) = 6.9999995 which floored to 6.
    // In f64, 1.0 / (2.0 / 14.0) = 7.0 which floors to 7.
    let expected = burn_tensor::TensorData::from([[[[0.0, 7.0]]]]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[cfg(feature = "flex")]
fn test_nearest_scale_driven_mapping() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data([[[[10., 20., 30., 40.]]]], &device);
    let output = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Nearest).with_scale_factor([1.0, 0.6]),
    );

    // The second coordinate is 1 / 0.6, which floors to 1; a size-derived scale would select 2.
    output.into_data().assert_eq(
        &burn_tensor::TensorData::from([[[[10., 20.]]]]).convert::<FloatElem>(),
        false,
    );
}

#[test]
fn test_nearest_presets() {
    let device = Default::default();
    let input = TestTensor::<4>::from_data([[[[10., 20., 30., 40.]]]], &device);
    for (mode, expected) in [
        (InterpolateMode::Nearest, [10., 30.]),
        (InterpolateMode::NearestExact, [20., 40.]),
    ] {
        let output = interpolate(
            input.clone(),
            InterpolateOptions::new(mode).with_output_size([1, 2]),
        );
        output.into_data().assert_eq(
            &burn_tensor::TensorData::from([[[expected]]]).convert::<FloatElem>(),
            false,
        );
    }
}

#[test]
fn test_nearest_asymmetric_14_to_18_ratio_precision() {
    let device = Default::default();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    // Shape [1, 1, 1, 14] -> [1, 1, 1, 18]
    let input = TestTensorInt::arange(0..14, &device)
        .float()
        .reshape([1, 1, 1, 14]);
    let output = burn_tensor::module::interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([1, 18]),
    );
    let values: Vec<f32> = output.into_data().try_into_vec_as::<f32>().unwrap();
    // Index 9 must map to 9 * 14 / 18 = 7.0 -> element 7 (not 6)
    assert_eq!(values[9], 7.0);
}

#[test]
fn test_nearest_exact_size_based_tie() {
    let options = InterpolateOptions::new(InterpolateMode::NearestExact);
    let device = Default::default();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    let input = TestTensor::<4>::from_data(
        [[[[
            11., 23., 37., 41., 53., 67., 79., 83., 97., 101., 113., 127., 139., 149.,
        ]]]],
        &device,
    );
    let output = interpolate(input, options.with_output_size([1, 9]));
    // Output index 4 maps exactly to 6.5; RoundPreferCeil must select input 7.
    let expected =
        burn_tensor::TensorData::from([[[[11., 37., 41., 67., 83., 97., 113., 127., 149.]]]]);
    output
        .into_data()
        .assert_eq(&expected.convert::<FloatElem>(), false);
}
