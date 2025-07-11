use crate::include_models;
include_models!(reduce_max, reduce_min, reduce_mean, reduce_prod, reduce_sum);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    fn reduce_max() {
        let device = Default::default();
        let model: reduce_max::Model<Backend> = reduce_max::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([25f32]);
        let expected = TensorData::from([[[[25f32]]]]);

        assert_eq!(output_scalar.to_data(), expected_scalar);
        assert_eq!(output_tensor.to_data(), input.to_data());
        assert_eq!(output_value.to_data(), expected);
    }

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
            .assert_approx_eq::<FT>(&expected_scalar, burn::tensor::Tolerance::default());
        output_tensor
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), burn::tensor::Tolerance::default());
        output_value
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_sum() {
        let device = Default::default();
        let model: reduce_sum::Model<Backend> = reduce_sum::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([39f32]);
        let expected = TensorData::from([[[[39f32]]]]);

        output_scalar.to_data().assert_eq(&expected_scalar, true);
        output_tensor.to_data().assert_eq(&input.to_data(), true);
        output_value.to_data().assert_eq(&expected, true);
    }
}