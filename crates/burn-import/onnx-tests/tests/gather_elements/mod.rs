// Import the shared macro
use crate::include_models;
include_models!(gather_elements);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

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
}
