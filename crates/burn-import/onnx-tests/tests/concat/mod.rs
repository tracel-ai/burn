// Import the shared macro
use crate::include_models;
include_models!(concat, concat_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn concat_tensors() {
        // Initialize the model
        let device = Default::default();
        let model: concat::Model<Backend> = concat::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::zeros([1, 2, 3, 5], &device);

        let output = model.forward(input);

        let expected = Shape::from([1, 18, 3, 5]);

        assert_eq!(output.shape(), expected);
    }

    #[test]
    fn concat_shapes() {
        // Initialize the model
        let device = Default::default();
        let model: concat_shape::Model<Backend> = concat_shape::Model::new(&device);

        // Create test inputs with the expected shapes
        let input1 = Tensor::<Backend, 2>::zeros([2, 3], &device);
        let input2 = Tensor::<Backend, 3>::zeros([4, 5, 6], &device);
        let input3 = Tensor::<Backend, 1>::zeros([7], &device);

        // Run the model - it extracts shapes and concatenates them
        let output = model.forward(input1, input2, input3);

        // The output should be an array [i64; 6] containing [2, 3, 4, 5, 6, 7]
        let expected: [i64; 6] = [2, 3, 4, 5, 6, 7];
        assert_eq!(output, expected);
    }
}
