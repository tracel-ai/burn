// Import the shared macro
use crate::include_models;
include_models!(equal);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn equal_scalar_to_scalar_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: equal::Model<Backend> = equal::Model::default();

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 1., 1., 1.]]]], &Default::default());

        let scalar = 2f64;
        let (tensor_out, scalar_out) = model.forward(input, scalar);

        #[cfg(feature = "bool-u32")]
        let expected_tensor = TensorData::from([[[[1u32, 1, 1, 1]]]]);

        #[cfg(not(feature = "bool-u32"))]
        let expected_tensor = TensorData::from([[[[true, true, true, true]]]]);

        tensor_out.to_data().assert_eq(&expected_tensor, true);

        let expected_scalar = false;
        assert_eq!(scalar_out, expected_scalar);
    }
}
