use crate::include_models;
include_models!(eye_like);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn eye_like_test() {
        // Test for EyeLike operation
        let device = Default::default();
        let model = eye_like::Model::<TestBackend>::new(&device);

        // Create a 3x3 input tensor (values don't matter for EyeLike, just the shape)
        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // Expected output is a 3x3 identity matrix
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_rectangular_test() {
        // Test for EyeLike operation with rectangular matrix
        let device = Default::default();

        // Create a 3x4 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([3, 4], &device);

        // For rectangular matrices, EyeLike should create identity in top-left corner
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            &device,
        );

        // We can use the same model since EyeLike adapts to input shape
        let model = eye_like::Model::<TestBackend>::new(&device);
        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
