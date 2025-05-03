// Include the models for this node type
use crate::include_models;
include_models!(add, add_int);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};
    

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn add_scalar_to_tensor_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add::Model<Backend> = add::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 2f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[9f32, 10., 11., 12.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_scalar_to_int_tensor_and_int_tensor_to_int_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add_int::Model<Backend> = add_int::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let scalar = 2;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[9i64, 11, 13, 15]]]]);

        output.to_data().assert_eq(&expected, true);
    }
}