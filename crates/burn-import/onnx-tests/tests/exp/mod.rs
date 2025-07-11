// Import the shared macro
use crate::include_models;
include_models!(exp);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    #[allow(clippy::approx_constant)]
    fn exp() {
        let device = Default::default();
        let model: exp::Model<Backend> = exp::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[0.0000, 0.6931]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1f32, 2.]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
