use crate::include_models;
include_models!(log);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    fn log() {
        let device = Default::default();
        let model: log::Model<Backend> = log::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0000f32, 1.3863, 2.1972, 3.2189]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::rel_abs(1e-4, 1e-4));
    }
}
