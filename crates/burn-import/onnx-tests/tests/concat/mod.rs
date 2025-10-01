// Import the shared macro
use crate::include_models;
include_models!(
    concat,
    concat_shape,
    concat_shape_with_constant,
    concat_mixed_single_element,
    concat_mixed_three_elements,
    concat_multiple_mixed,
    concat_with_constants
);

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

    #[test]
    fn concat_mixed_single_element() {
        // Initialize the model
        let device = Default::default();
        let model: concat_mixed_single_element::Model<TestBackend> =
            concat_mixed_single_element::Model::new(&device);

        // Create test input with shape [2, 3]
        let input1 = Tensor::<TestBackend, 2>::zeros([2, 3], &device);

        // Run the model - it extracts shape and concatenates with constant [100]
        let output = model.forward(input1);

        // The output should be an array [i64; 3] containing [2, 3, 100]
        let expected: [i64; 3] = [2, 3, 100];
        assert_eq!(output, expected);
    }

    #[test]
    fn concat_mixed_three_elements() {
        // Initialize the model
        let device = Default::default();
        let model: concat_mixed_three_elements::Model<TestBackend> =
            concat_mixed_three_elements::Model::new(&device);

        // Create test input with shape [4, 5, 6]
        let input1 = Tensor::<TestBackend, 3>::zeros([4, 5, 6], &device);

        // Run the model - it extracts shape and concatenates with constant [10, 20, 30]
        let output = model.forward(input1);

        // The output should be an array [i64; 6] containing [4, 5, 6, 10, 20, 30]
        let expected: [i64; 6] = [4, 5, 6, 10, 20, 30];
        assert_eq!(output, expected);
    }

    #[test]
    fn concat_multiple_mixed() {
        // Initialize the model
        let device = Default::default();
        let model: concat_multiple_mixed::Model<TestBackend> =
            concat_multiple_mixed::Model::new(&device);

        // Create test inputs
        let input1 = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
        let input2 = Tensor::<TestBackend, 3>::zeros([4, 5, 6], &device);

        // Run the model - it concatenates shapes and constants
        let output = model.forward(input1, input2);

        // The output should be an array [i64; 8] containing [2, 3, 100, 200, 4, 5, 6, 300]
        let expected: [i64; 8] = [2, 3, 100, 200, 4, 5, 6, 300];
        assert_eq!(output, expected);
    }

    #[test]
    fn concat_with_constants() {
        // Initialize the model
        let device = Default::default();
        // Use Model::default() to load constants from the record file
        let model: concat_with_constants::Model<TestBackend> =
            concat_with_constants::Model::default();

        // Create test input with shape [3, 4]
        let input1 = Tensor::<TestBackend, 2>::zeros([3, 4], &device);

        // Run the model - it concatenates shape with multiple constant tensors
        let output = model.forward(input1);

        // The output should be [3, 4, 2, 3, 5, 7, 8, 9]
        // Shape: [3, 4] + const1: [2, 3] + const2: [5] + const3: [7, 8, 9]
        let expected: [i64; 8] = [3, 4, 2, 3, 5, 7, 8, 9];
        assert_eq!(output, expected);
    }
}
