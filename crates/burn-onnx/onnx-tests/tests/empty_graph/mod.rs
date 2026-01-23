// Import the shared macro
use crate::include_models;
include_models!(
    empty_graph_scalar,
    empty_graph_scalar_int,
    empty_graph_shape,
    empty_graph_tensor,
    empty_graph_multiple
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn empty_graph_scalar_test() {
        // Test that a float scalar input is returned unchanged
        let device = Default::default();
        let model = empty_graph_scalar::Model::<TestBackend>::new(&device);

        // Input scalar
        let input: f32 = 42.0;
        let output: f32 = model.forward(input);

        // Output should equal input
        assert_eq!(output, input);
    }

    #[test]
    fn empty_graph_scalar_int_test() {
        // Test that an int64 scalar input is returned unchanged
        let device = Default::default();
        let model = empty_graph_scalar_int::Model::<TestBackend>::new(&device);

        // Input scalar
        let input: i64 = 123;
        let output: i64 = model.forward(input);

        // Output should equal input
        assert_eq!(output, input);
    }

    #[test]
    fn empty_graph_shape_test() {
        // Test that a shape tensor input is returned unchanged
        let device = Default::default();
        let model = empty_graph_shape::Model::<TestBackend>::new(&device);

        // Input shape tensor [2, 3, 4]
        let input = Tensor::<TestBackend, 1, Int>::from_data([2i64, 3i64, 4i64], &device);
        let output = model.forward(input.clone());

        // Output should equal input
        output.to_data().assert_eq(&input.to_data(), true);
    }

    #[test]
    fn empty_graph_tensor_test() {
        // Test that a tensor input is returned unchanged
        let device = Default::default();
        let model = empty_graph_tensor::Model::<TestBackend>::new(&device);

        // Input tensor [2, 3]
        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let output = model.forward(input.clone());

        // Output should equal input
        output.to_data().assert_eq(&input.to_data(), true);
    }

    #[test]
    fn empty_graph_multiple_test() {
        // Test that multiple inputs are returned unchanged
        let device = Default::default();
        let model = empty_graph_multiple::Model::<TestBackend>::new(&device);

        // Input 1: shape tensor
        let input1 = Tensor::<TestBackend, 1, Int>::from_data([5i64, 6i64, 7i64], &device);
        // Input 2: tensor
        let input2 =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        let (output1, output2) = model.forward(input1.clone(), input2.clone());

        // All outputs should equal their respective inputs
        output1.to_data().assert_eq(&input1.to_data(), true);
        output2.to_data().assert_eq(&input2.to_data(), true);
    }
}
