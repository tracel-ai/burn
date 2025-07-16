// Import the shared macro
use crate::include_models;
include_models!(concat);

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
}
