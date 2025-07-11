// Import the shared macro
use crate::include_models;
include_models!(div);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn div_tensor_by_scalar_and_tensor_by_tensor() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: div::Model<Backend> = div::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[3., 6., 6., 9.]]]], &device);
        let scalar1 = 9.0f64;
        let scalar2 = 3.0f64;
        let output = model.forward(input, scalar1, scalar2);
        let expected = TensorData::from([[[[1f32, 2., 2., 3.]]]]);

        output.to_data().assert_eq(&expected, true);
    }
}
