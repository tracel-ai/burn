use burn_core::tensor::Tolerance;
use burn_vision::{
    BorderType, KernelShape, MorphOptions, Morphology, Point, Size, create_structuring_element,
};
type FT = f32;

mod common;
use common::*;

#[test]
fn should_support_dilate_luma() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(5, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_5x5_Rect.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_luma_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(5, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_5x5_Cross.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_luma_ellipse() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Ellipse, Size::new(5, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_5x5_Ellipse.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_luma_non_square_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(3, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_3x5_Rect.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_luma_non_square_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(3, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_3x5_Cross.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(3, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_2_3x5_Rect.png", &device, false);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(3, 5), None, &device);

    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_2_3x5_Cross.png", &device, false);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_border_reflect_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(7, 7), None, &device);

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .border_type(BorderType::Reflect)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_7x7_Rect_BORDER_REFLECT.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_border_reflect_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(7, 7), None, &device);

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .border_type(BorderType::Reflect)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_7x7_Cross_BORDER_REFLECT.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_border_reflect101_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(7, 7), None, &device);

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .border_type(BorderType::Reflect101)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_7x7_Rect_BORDER_REFLECT101.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_border_reflect101_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(7, 7), None, &device);

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .border_type(BorderType::Reflect101)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_7x7_Cross_BORDER_REFLECT101.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_border_replicate_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(7, 7), None, &device);

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .border_type(BorderType::Replicate)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_7x7_Rect_BORDER_REPLICATE.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_border_replicate_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(7, 7), None, &device);

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .border_type(BorderType::Replicate)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_7x7_Cross_BORDER_REPLICATE.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_anchor_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(
        KernelShape::Rect,
        Size::new(5, 7),
        Some(Point::new(1, 2)),
        &device,
    );

    let output = tensor.dilate(
        kernel,
        MorphOptions::builder().anchor(Point::new(2, 1)).build(),
    );
    let expected = test_image("morphology/Dilate_2_5x7_Rect_ANCHOR.png", &device, false);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_rgb_anchor_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_2.png", &device, false);
    let kernel = create_structuring_element(
        KernelShape::Cross,
        Size::new(5, 7),
        Some(Point::new(1, 2)),
        &device,
    );

    // With default border, bottom left pixel is undefined with this particular kernel and anchor
    // Use replicate instead for comparability
    let output = tensor.dilate(
        kernel,
        MorphOptions::builder()
            .anchor(Point::new(2, 1))
            .border_type(BorderType::Replicate)
            .build(),
    );
    let expected = test_image(
        "morphology/Dilate_2_5x7_Cross_ANCHOR_BORDER_REPLICATE.png",
        &device,
        false,
    );
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_boolean_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true).greater_elem(0);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(5, 5), None, &device);

    // With default border, bottom left pixel is undefined with this particular kernel and anchor
    // Use replicate instead for comparability
    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_5x5_Rect.png", &device, true).greater_elem(0);
    let expected = TestTensorBool::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_boolean_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true).greater_elem(0);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(5, 5), None, &device);

    // With default border, bottom left pixel is undefined with this particular kernel and anchor
    // Use replicate instead for comparability
    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = test_image("morphology/Dilate_1_5x5_Cross.png", &device, true).greater_elem(0);
    let expected = TestTensorBool::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_int_rect() {
    let device = TestDevice::default().into();
    let tensor = (test_image("morphology/Base_1.png", &device, true) * 255.0).int();
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(5, 5), None, &device);

    // With default border, bottom left pixel is undefined with this particular kernel and anchor
    // Use replicate instead for comparability
    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = (test_image("morphology/Dilate_1_5x5_Rect.png", &device, true) * 255.0).int();
    let expected = TestTensorInt::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_dilate_int_cross() {
    let device = TestDevice::default().into();
    let tensor = (test_image("morphology/Base_1.png", &device, true) * 255.0).int();
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(5, 5), None, &device);

    // With default border, bottom left pixel is undefined with this particular kernel and anchor
    // Use replicate instead for comparability
    let output = tensor.dilate(kernel, MorphOptions::default());
    let expected = (test_image("morphology/Dilate_1_5x5_Cross.png", &device, true) * 255.0).int();
    let expected = TestTensorInt::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_erode_luma() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = TestTensorBool::<2>::from([
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
    ]);

    let output = tensor.erode(kernel, MorphOptions::default());
    let expected = test_image("morphology/Erode_1_5x5_Rect.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_erode_luma_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(5, 5), None, &device);

    let output = tensor.erode(kernel, MorphOptions::default());
    let expected = test_image("morphology/Erode_1_5x5_Cross.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn should_support_erode_luma_ellipse() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Ellipse, Size::new(5, 5), None, &device);

    let output = tensor.erode(kernel, MorphOptions::default());
    let expected = test_image("morphology/Erode_1_5x5_Ellipse.png", &device, true);
    let expected = Tensor::<3>::from(expected);

    output
        .into_data()
        .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::absolute(1e-6));
}

#[test]
fn create_structuring_element_should_match_manual_rect() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Rect, Size::new(5, 5), None, &device);
    let kernel_manual = TestTensorBool::<2>::from([
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
    ]);

    let output = tensor.clone().dilate(kernel, MorphOptions::default());
    let output_manual = tensor.dilate(kernel_manual, MorphOptions::default());

    output
        .into_data()
        .assert_eq(&output_manual.into_data(), false);
}

#[test]
fn create_structuring_element_should_match_manual_cross() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Cross, Size::new(5, 5), None, &device);
    let kernel_manual = TestTensorBool::<2>::from([
        [false, false, true, false, false],
        [false, false, true, false, false],
        [true, true, true, true, true],
        [false, false, true, false, false],
        [false, false, true, false, false],
    ]);

    let output = tensor.clone().dilate(kernel, MorphOptions::default());
    let output_manual = tensor.dilate(kernel_manual, MorphOptions::default());

    output
        .into_data()
        .assert_eq(&output_manual.into_data(), false);
}
#[test]
fn create_structuring_element_should_match_manual_ellipse() {
    let device = TestDevice::default().into();
    let tensor = test_image("morphology/Base_1.png", &device, true);
    let kernel = create_structuring_element(KernelShape::Ellipse, Size::new(5, 5), None, &device);
    let kernel_manual = TestTensorBool::<2>::from([
        [false, false, true, false, false],
        [true, true, true, true, true],
        [true, true, true, true, true],
        [true, true, true, true, true],
        [false, false, true, false, false],
    ]);

    let output = tensor.clone().dilate(kernel, MorphOptions::default());
    let output_manual = tensor.dilate(kernel_manual, MorphOptions::default());

    output
        .into_data()
        .assert_eq(&output_manual.into_data(), false);
}
