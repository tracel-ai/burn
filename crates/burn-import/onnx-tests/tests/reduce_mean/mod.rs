// Import the shared macro
use crate::include_models;
include_models!(reduce_mean);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn reduce_mean() {
        let device = Default::default();
        let model: reduce_mean::Model<Backend> = reduce_mean::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([9.75f32]);
        let expected = TensorData::from([[[[9.75f32]]]]);

        output_scalar.to_data().assert_eq(&expected_scalar, true);
        output_tensor.to_data().assert_eq(&input.to_data(), true);
        output_value.to_data().assert_eq(&expected, true);
    }
}
