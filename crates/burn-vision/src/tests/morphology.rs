#[burn_tensor_testgen::testgen(morphology)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use burn_tensor::TensorData;
    use burn_vision::{
        as_type, create_structuring_element,
        tests::{save_test_image, test_image},
        KernelShape, MorphOptions, Morphology,
    };

    #[test]
    fn should_support_dilate_luma() {
        let tensor = test_image("morphology/Base_1.png", &Default::default(), true);
        let kernel = create_structuring_element::<TestBackend>(
            KernelShape::Rect,
            5,
            5,
            None,
            &Default::default(),
        );

        let output = tensor.dilate(kernel, MorphOptions::default());
        let expected = test_image(
            "morphology/Dilate_1_5x5_Rect.png",
            &Default::default(),
            true,
        );
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_dilate_rgb() {
        let tensor = test_image("morphology/Base_1.png", &Default::default(), false);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.dilate(kernel, MorphOptions::default());
        let expected = test_image(
            "morphology/Dilate_1_5x5_Rect.png",
            &Default::default(),
            false,
        );
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_erode_luma() {
        let tensor = test_image("morphology/Base_1.png", &Default::default(), true);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.erode(kernel, MorphOptions::default());
        let expected = test_image("morphology/Erode_1_5x5_Rect.png", &Default::default(), true);
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_erode_rgb() {
        let tensor = test_image("morphology/Base_1.png", &Default::default(), false);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.erode(kernel, MorphOptions::default());
        let expected = test_image(
            "morphology/Erode_1_5x5_Rect.png",
            &Default::default(),
            false,
        );
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn create_structuring_element_should_match_manual_rect() {
        let tensor = test_image("morphology/Base_1.png", &Default::default(), true);
        let kernel = create_structuring_element::<TestBackend>(
            KernelShape::Rect,
            5,
            5,
            None,
            &Default::default(),
        );
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
        let tensor = test_image("morphology/Base_1.png", &Default::default(), true);
        let kernel = create_structuring_element::<TestBackend>(
            KernelShape::Cross,
            5,
            5,
            None,
            &Default::default(),
        );
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
        let tensor = test_image("morphology/Base_1.png", &Default::default(), true);
        let kernel = create_structuring_element::<TestBackend>(
            KernelShape::Ellipse,
            5,
            5,
            None,
            &Default::default(),
        );
        let kernel_manual = TestTensorBool::<2>::from([
            [false, false, true, false, false],
            [false, true, true, true, false],
            [true, true, true, true, true],
            [false, true, true, true, false],
            [false, false, true, false, false],
        ]);

        let output = tensor.clone().dilate(kernel, MorphOptions::default());
        let output_manual = tensor.dilate(kernel_manual, MorphOptions::default());

        output
            .into_data()
            .assert_eq(&output_manual.into_data(), false);
    }
}
