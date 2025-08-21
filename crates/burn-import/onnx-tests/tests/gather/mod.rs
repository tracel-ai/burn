// Import the shared macro
use crate::include_models;
include_models!(
    gather_1d_idx,
    gather_2d_idx,
    gather_elements,
    gather_scalar,
    gather_scalar_out,
    gather_shape,
    gather_with_shape_indices
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn gather_1d_idx() {
        let model: gather_1d_idx::Model<TestBackend> = gather_1d_idx::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = Tensor::<TestBackend, 1, Int>::from_ints([0, 2], &device);
        let expected = TensorData::from([[1f32, 3.], [4., 6.]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_2d_idx() {
        let model: gather_2d_idx::Model<TestBackend> = gather_2d_idx::Model::default();

        let device = Default::default();

        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], &device);
        let index = Tensor::<TestBackend, 2, Int>::from_data([[0, 1], [1, 2]], &device);
        let expected = TensorData::from([[[1f32, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_shape() {
        let model: gather_shape::Model<TestBackend> = gather_shape::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        // shape(input) = [2, 3]
        let index = Tensor::<TestBackend, 1, Int>::from_ints([0], &device);
        let output = model.forward(input, index);
        let expected = [2i64];

        assert_eq!(output, expected);
    }

    #[test]
    fn gather_scalar() {
        let model: gather_scalar::Model<TestBackend> = gather_scalar::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = 0;
        let output = model.forward(input, index);
        let expected = TensorData::from([1f32, 2., 3.]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_scalar_out() {
        let model: gather_scalar_out::Model<TestBackend> = gather_scalar_out::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats([1., 2., 3.], &device);
        let index = 1;
        let output = model.forward(input, index);

        let expected = 2.0f32;
        assert_eq!(output, expected);
    }

    #[test]
    fn gather_elements() {
        // Initialize the model with weights (loaded from the exported file)
        let model: gather_elements::Model<TestBackend> = gather_elements::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let index = Tensor::<TestBackend, 2, Int>::from_ints([[0, 0], [1, 0]], &device);
        let output = model.forward(input, index);
        let expected = TensorData::from([[1f32, 1.], [4., 3.]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_with_shape_indices() {
        // Test the most comprehensive case of our runtime Shape indices implementation:
        // This is the exact scenario that was causing the original panic and required our full fix.
        let model: gather_with_shape_indices::Model<TestBackend> =
            gather_with_shape_indices::Model::default();

        let device = Default::default();

        // Input tensor with shape [2, 3]
        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input);

        // Expected: shape [2, 3] used as indices to gather from [100, 200, 300, 400, 500]
        // Gathering at indices [2, 3] should give us [300, 400]
        let expected = TensorData::from([300i64, 400]);

        assert_eq!(output.to_data(), expected);
    }
}
