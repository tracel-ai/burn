// Import the shared macro
use crate::include_models;
include_models!(reduce_min);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn reduce_min() {
        let device = Default::default();
        let model: reduce_min::Model<Backend> = reduce_min::Model::new(&device);

        // Run the models
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([1f32]);
        let expected = TensorData::from([[[[1f32]]]]);

        assert_eq!(output_scalar.to_data(), expected_scalar);
        assert_eq!(output_tensor.to_data(), input.to_data());
        assert_eq!(output_value.to_data(), expected);
    }
}
