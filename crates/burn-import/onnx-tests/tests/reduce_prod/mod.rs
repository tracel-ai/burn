// Import the shared macro
use crate::include_models;
include_models!(reduce_prod);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::Backend;
    type FT = FloatElem<Backend>;

    #[test]
    fn reduce_prod() {
        let device = Default::default();
        let model: reduce_prod::Model<Backend> = reduce_prod::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([900f32]);
        let expected = TensorData::from([[[[900f32]]]]);

        // Tolerance of 0.001 since floating-point multiplication won't be perfect
        output_scalar
            .to_data()
            .assert_approx_eq::<FT>(&expected_scalar, Tolerance::default());
        output_tensor
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
        output_value
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
