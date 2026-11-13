use super::*;
use burn_tensor::module::interpolate3d;
use burn_tensor::ops::{CoordinateTransformMode, Interpolate3dMode, Interpolate3dOptions};
use burn_tensor::{TensorData, Tolerance};

// References: torch.nn.functional.interpolate with float64 inputs, followed by
// (output * upstream).sum().backward(). Each fixture uses nonuniform upstream gradients.

fn assert_nearest_backward(mode: Interpolate3dMode, expected: [f32; 8]) {
    let device = AutodiffDevice::new();
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            vec![2.0f32, -3.0, 7.0, 1.0, -5.0, 9.0, 4.0, 11.0],
            [1, 1, 4, 1, 2],
        ),
        &device,
    )
    .require_grad();
    let output = interpolate3d(
        input.clone(),
        Interpolate3dOptions::new(mode).with_output_size([2, 1, 3]),
    );
    let upstream = TestTensor::<5>::from_data(
        TensorData::new(vec![2.0f32, -1.0, 3.0, 5.0, -2.0, 4.0], [1, 1, 2, 1, 3]),
        &device,
    );
    let grads = (output * upstream).sum().backward();
    input.grad(&grads).unwrap().into_data().assert_eq(
        &TensorData::new(expected.to_vec(), [1, 1, 4, 1, 2]).convert::<FloatElem>(),
        false,
    );
}

#[test]
fn nearest_backward_scatter() {
    assert_nearest_backward(
        Interpolate3dMode::Nearest,
        [1.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
    );
}

#[test]
fn nearest_exact_backward_scatter() {
    assert_nearest_backward(
        Interpolate3dMode::NearestExact,
        [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 5.0, 2.0],
    );
}

fn assert_trilinear_backward(coordinate_transform: CoordinateTransformMode, expected: [f32; 24]) {
    let device = AutodiffDevice::new();
    // The reference leaf has shape (1, 2, 2, 3, 2); interpolate its permuted (1, 2, 2, 2, 3) view.
    let input = TestTensor::<5>::from_data(
        TensorData::new(
            (0..24).map(|i| (i * 7 % 19) as f32 - 9.0).collect(),
            [1, 2, 2, 3, 2],
        ),
        &device,
    )
    .require_grad();
    let output = interpolate3d(
        input.clone().permute([0, 1, 2, 4, 3]),
        Interpolate3dOptions::new(Interpolate3dMode::Trilinear {
            coordinate_transform,
        })
        .with_output_size([3, 1, 2]),
    );
    let upstream = TestTensor::<5>::from_data(
        TensorData::new(
            vec![
                2.0f32, -1.0, 3.0, 5.0, -2.0, 4.0, 1.0, 6.0, -3.0, 2.0, 4.0, -5.0,
            ],
            [1, 2, 3, 1, 2],
        ),
        &device,
    );
    let grads = (output * upstream).sum().backward();
    input
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::new(expected.to_vec(), [1, 2, 2, 3, 2]),
            Tolerance::default(),
        );
}

#[test]
fn trilinear_half_pixel_backward_permuted_channels() {
    assert_trilinear_backward(
        CoordinateTransformMode::HalfPixel,
        [
            1.3125, 1.3125, 0.625, 0.625, 0.5625, 0.5625, -0.1875, -0.1875, 0.75, 0.75, 2.4375,
            2.4375, -0.1875, -0.1875, 0.8125, 0.8125, 2.625, 2.625, 0.9375, 0.9375, -0.1875,
            -0.1875, -1.5, -1.5,
        ],
    );
}

#[test]
fn trilinear_align_corners_backward_permuted_channels() {
    assert_trilinear_backward(
        CoordinateTransformMode::AlignCorners,
        [
            3.5, 0.0, 0.0, 0.0, 1.5, 0.0, -0.5, 0.0, 0.0, 0.0, 6.5, 0.0, -0.5, 0.0, 0.0, 0.0, 7.0,
            0.0, 2.5, 0.0, 0.0, 0.0, -4.0, 0.0,
        ],
    );
}

#[test]
fn interpolate3d_empty_input_keeps_gradient_connected() {
    let device = AutodiffDevice::new();
    let input = TestTensor::<5>::empty([0, 2, 2, 3, 4], &device).require_grad();
    let output = interpolate3d(
        input.clone(),
        Interpolate3dOptions::new(Interpolate3dMode::Nearest).with_output_size([3, 1, 2]),
    );
    let grads = output.backward();
    input.grad(&grads).unwrap().into_data().assert_eq(
        &TensorData::new(Vec::<FloatElem>::new(), [0, 2, 2, 3, 4]),
        false,
    );
}
