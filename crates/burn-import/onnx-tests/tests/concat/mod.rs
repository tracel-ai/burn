// Import the shared macro
use crate::include_models;
include_models!(concat, concat_shape, concat_shape_with_constant);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn concat_tensors() {
        // Initialize the model
        let device = Default::default();
        let model: concat::Model<TestBackend> = concat::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::zeros([1, 2, 3, 5], &device);

        let output = model.forward(input);

        let expected = Shape::from([1, 18, 3, 5]);

        assert_eq!(output.shape(), expected);
    }

    #[test]
    fn concat_shapes() {
        // Initialize the model
        let device = Default::default();
        let model: concat_shape::Model<TestBackend> = concat_shape::Model::new(&device);

        // Create test inputs with the expected shapes
        let input1 = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
        let input2 = Tensor::<TestBackend, 3>::zeros([4, 5, 6], &device);
        let input3 = Tensor::<TestBackend, 1>::zeros([7], &device);

        // Run the model - it extracts shapes and concatenates them
        let output = model.forward(input1, input2, input3);

        // The output should be an array [i64; 6] containing [2, 3, 4, 5, 6, 7]
        let expected: [i64; 6] = [2, 3, 4, 5, 6, 7];
        assert_eq!(output, expected);
    }

    #[test]
    fn concat_shape_with_constant() {
        // Initialize the model
        let device = Default::default();
        let model: concat_shape_with_constant::Model<TestBackend> =
            concat_shape_with_constant::Model::new(&device);

        // Create test input with shape [3, 4, 5]
        let input1 = Tensor::<TestBackend, 3>::zeros([3, 4, 5], &device);

        // Run the model - it extracts shape and concatenates with constant [10, 20]
        let output = model.forward(input1);

        // The output should be an array [i64; 5] containing [3, 4, 5, 10, 20]
        let expected: [i64; 5] = [3, 4, 5, 10, 20];
        assert_eq!(output, expected);
    }
}
