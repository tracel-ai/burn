use burn_core::Tensor;
use burn_std::{Shape, TensorData, Tolerance};
use burn_vision::ColorConversion;

mod common;
use common::*;

fn pixel_to_tensor(rgb: [f32; 3]) -> Tensor<4> {
    Tensor::<4>::from_data(
        TensorData::new(rgb.to_vec(), Shape::new([1, 3, 1, 1])),
        &TestDevice::default().into(),
    )
}

fn gray_to_tensor(gray: f32) -> Tensor<4> {
    Tensor::<4>::from_data(
        TensorData::new(vec![gray], Shape::new([1, 1, 1, 1])),
        &TestDevice::default().into(),
    )
}

fn assert_gray(rgb: [f32; 3], expected: f32) {
    pixel_to_tensor(rgb)
        .rgb2gray()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(vec![expected], Shape::new([1, 1, 1, 1])),
            Tolerance::default(),
        );
}

fn assert_rgb_from_gray(gray: f32, expected: &[f32; 3]) {
    gray_to_tensor(gray)
        .gray2rgb()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(expected.to_vec(), Shape::new([1, 3, 1, 1])),
            Tolerance::default(),
        );
}

fn assert_hsv(rgb: [f32; 3], expected: &[f32; 3]) {
    pixel_to_tensor(rgb)
        .rgb2hsv()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(expected.to_vec(), Shape::new([1, 3, 1, 1])),
            Tolerance::default(),
        );
}

fn assert_rgb(hsv: [f32; 3], expected: &[f32; 3]) {
    pixel_to_tensor(hsv)
        .hsv2rgb()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(expected.to_vec(), Shape::new([1, 3, 1, 1])),
            Tolerance::default(),
        );
}

#[test]
fn test_rgb2hsv() {
    // Expected values computed with OpenCV.
    assert_hsv([0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
    assert_hsv([1.0, 0.0, 0.0], &[0.0, 1.0, 1.0]);
    assert_hsv([0.0, 1.0, 0.0], &[120.0, 1.0, 1.0]);
    assert_hsv([0.0, 0.0, 1.0], &[240.0, 1.0, 1.0]);
    assert_hsv([1.0, 1.0, 1.0], &[0.0, 0.0, 1.0]);
    assert_hsv([0.5, 0.5, 0.5], &[0.0, 0.0, 0.5]);
    assert_hsv([0.2, 0.6, 0.9], &[205.7143, 0.7778, 0.9000]);
    assert_hsv([0.78, 0.0, 0.45], &[325.3846, 1.0000, 0.7800]);
}

#[test]
fn test_hsv2rgb() {
    // Expected values computed with OpenCV.
    assert_rgb([0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
    assert_rgb([0.0, 1.0, 1.0], &[1.0, 0.0, 0.0]);
    assert_rgb([120.0, 1.0, 1.0], &[0.0, 1.0, 0.0]);
    assert_rgb([240.0, 1.0, 1.0], &[0.0, 0.0, 1.0]);
    assert_rgb([0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]);
    assert_rgb([0.0, 0.5, 1.0], &[1.0, 0.5, 0.5]);
    assert_rgb([180.0, 0.6, 0.9], &[0.36, 0.9, 0.9]);
    assert_rgb([273.0, 0.17, 0.42], &[0.3879, 0.3486, 0.42]);
}

#[test]
fn rgb_hsv_roundtrip() {
    let expected = Tensor::cat(
        vec![
            pixel_to_tensor([0.8, 0.3, 0.1]),
            pixel_to_tensor([0.1, 0.7, 0.4]),
            pixel_to_tensor([0.2, 0.2, 0.9]),
            pixel_to_tensor([0.0, 0.0, 0.0]),
            pixel_to_tensor([1.0, 1.0, 1.0]),
        ],
        3,
    );
    let actual = expected.clone().rgb2hsv().hsv2rgb();
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
}

#[test]
fn test_rgb2gray() {
    assert_gray([0.0, 0.0, 0.0], 0.0);
    assert_gray([1.0, 0.0, 0.0], 0.299);
    assert_gray([0.0, 1.0, 0.0], 0.587);
    assert_gray([0.0, 0.0, 1.0], 0.114);
    assert_gray([1.0, 1.0, 1.0], 1.0);
    assert_gray([0.5, 0.5, 0.5], 0.5);
    assert_gray([0.2, 0.6, 0.9], 0.5146);
}

#[test]
fn test_gray2rgb() {
    assert_rgb_from_gray(0.0, &[0.0, 0.0, 0.0]);
    assert_rgb_from_gray(0.42, &[0.42, 0.42, 0.42]);
    assert_rgb_from_gray(1.0, &[1.0, 1.0, 1.0]);
}

#[test]
fn gray_rgb_roundtrip() {
    // The luminance weights sum to 1, so gray -> rgb -> gray is the identity.
    let gray = Tensor::<4>::from_data(
        TensorData::new(vec![0.0, 0.25, 0.5, 0.75, 1.0], Shape::new([1, 1, 1, 5])),
        &TestDevice::default().into(),
    );
    let actual = gray.clone().gray2rgb().rgb2gray();
    actual
        .into_data()
        .assert_approx_eq::<f32>(&gray.into_data(), Tolerance::default());
}
