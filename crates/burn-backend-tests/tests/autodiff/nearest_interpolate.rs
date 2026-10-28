use super::*;
use burn_tensor::Shape;
use burn_tensor::Tolerance;
use burn_tensor::module::interpolate;
use burn_tensor::ops::{CoordinateTransformMode, InterpolateMode, InterpolateOptions};

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
        let device = AutodiffDevice::new();
        let x = TestTensor::from_data(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &x_grad.device())
                .reshape::<4, _>(shape_x)
                .into_data(),
            &device,
        )
        .require_grad();

        let output = interpolate(
            x.clone(),
            InterpolateOptions::new(InterpolateMode::Nearest)
                .with_output_size([self.height_out, self.width_out]),
        );

        let grads = output.backward();
        let x_grad_actual = x.grad(&grads).unwrap();

        x_grad
            .to_data()
            .assert_approx_eq::<FloatElem>(&x_grad_actual.into_data(), Tolerance::permissive());
    }
}

#[test]
fn test_nearest_downsample_f64_precision_backward() {
    let device = AutodiffDevice::new();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    let input_vec: [f32; 14] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let x = TestTensor::<4>::from_data([[[input_vec]]], &device).require_grad();

    let output = interpolate(
        x.clone(),
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([1, 2]),
    );

    let grads = output.backward();
    let x_grad = x.grad(&grads).unwrap();

    let mut expected_grad = [0.0f32; 14];
    expected_grad[0] = 1.0;
    expected_grad[7] = 1.0;
    let expected = TestTensor::<4>::from([[[expected_grad]]]);
    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad.into_data(), Tolerance::default());
}

#[test]
fn test_nearest_asymmetric_14_to_18_backward_scatter() {
    let device = AutodiffDevice::new();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    let input = TestTensorInt::arange(0..14, &device)
        .float()
        .reshape([1, 1, 1, 14])
        .require_grad();
    let output = burn_tensor::module::interpolate(
        input.clone(),
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([1, 18]),
    );
    let grad = output.backward();
    let input_grad = input.grad(&grad).unwrap();
    let grad_values: Vec<f32> = input_grad.into_data().try_into_vec_as::<f32>().unwrap();
    assert_eq!(
        grad_values,
        [2., 1., 1., 2., 1., 1., 1., 2., 1., 1., 2., 1., 1., 1.]
    );
}

#[test]
fn test_bilinear_backward_non_integer_scale_gradients() {
    let device = AutodiffDevice::new();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    let input = TestTensor::<4>::from_data([[[[11.0, 23.0, 37.0, 41.0]]]], &device).require_grad();
    let options = InterpolateOptions::new(InterpolateMode::Bilinear {
        coordinate_transform: CoordinateTransformMode::HalfPixel,
    })
    .with_scale_factor([1.0, 0.6]);

    let output = interpolate(input.clone(), options);
    let grad = output.backward();
    // A scale of 0.6 samples coordinates 1/3 and 2, unlike the size ratio 2/4.
    let expected = burn_tensor::TensorData::from([[[[2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0]]]]);
    input
        .grad(&grad)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_bicubic_backward_custom_coefficient_clamps_boundary() {
    let device = AutodiffDevice::new();
    if !format!("{device:?}").to_lowercase().contains("flex") {
        return;
    }
    let input = TestTensor::<4>::from_data([[[[11.0, 23.0, 37.0, 41.0]]]], &device).require_grad();
    let options = InterpolateOptions::new(InterpolateMode::Bicubic {
        coordinate_transform: CoordinateTransformMode::Asymmetric,
        a: -0.5,
    })
    .with_scale_factor([1.0, 2.0]);

    let output = interpolate(input.clone(), options);
    let grad = output.slice([0..1, 0..1, 0..1, 1..2]).backward();
    // At coordinate 0.5 the weights are [-1, 9, 9, -1] / 16; the first two taps clamp to input 0.
    let expected = burn_tensor::TensorData::from([[[[8.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0, 0.0]]]]);
    input
        .grad(&grad)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_nearest_exact_backward() {
    let device = AutodiffDevice::new();
    let input = TestTensor::<4>::from_data([[[[10., 20., 30., 40.]]]], &device).require_grad();
    let output = interpolate(
        input.clone(),
        InterpolateOptions::new(InterpolateMode::NearestExact).with_output_size([1, 2]),
    );
    let grads = output.backward();
    input.grad(&grads).unwrap().into_data().assert_eq(
        &burn_tensor::TensorData::from([[[[0., 1., 0., 1.]]]]).convert::<FloatElem>(),
        false,
    );
}
