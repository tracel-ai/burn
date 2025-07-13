// Import the shared macro
use crate::include_models;
include_models!(mul);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn mul_scalar_with_tensor_and_tensor_with_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: mul::Model<Backend> = mul::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 6.0f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[126f32, 252., 378., 504.]]]]);

        output.to_data().assert_eq(&expected, true);
    }
}
