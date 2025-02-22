#[burn_tensor_testgen::testgen(morphology)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use burn_tensor::TensorData;
    use burn_vision::{
        as_type,
        tests::{save_test_image, test_image},
        Morphology,
    };

    #[test]
    fn should_support_dilate_luma() {
        let tensor = test_image("Morphology_1_Base.png", &Default::default(), true);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.dilate(kernel);
        let expected = test_image("Morphology_1_Dilation.png", &Default::default(), true);
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_dilate_rgb() {
        let tensor = test_image("Morphology_1_Base.png", &Default::default(), false);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.dilate(kernel);
        let expected = test_image("Morphology_1_Dilation.png", &Default::default(), false);
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_erode_luma() {
        let tensor = test_image("Morphology_1_Base.png", &Default::default(), true);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.erode(kernel);
        let expected = test_image("Morphology_1_Erosion.png", &Default::default(), true);
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_erode_rgb() {
        let tensor = test_image("Morphology_1_Base.png", &Default::default(), false);
        let kernel = TestTensorBool::<2>::from([
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
            [true, true, true, true, true],
        ]);

        let output = tensor.erode(kernel);
        let expected = test_image("Morphology_1_Erosion.png", &Default::default(), false);
        let expected = TestTensor::<3>::from(expected);

        output.into_data().assert_eq(&expected.into_data(), false);
    }
}
