use crate::include_models;
include_models!(flatten, flatten_2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn flatten() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: flatten::Model<Backend> = flatten::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 3>::ones([1, 5, 15], &device);
        let output = model.forward(input);

        let expected_shape = Shape::from([1, 75]);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn flatten_2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: flatten_2d::Model<Backend> = flatten_2d::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::ones([2, 3, 4, 5], &device);
        let output = model.forward(input);

        // Flatten leading and trailing dimensions (axis = 2) and returns a 2D tensor
        let expected_shape = Shape::from([6, 20]);
        assert_eq!(expected_shape, output.shape());
    }
}
