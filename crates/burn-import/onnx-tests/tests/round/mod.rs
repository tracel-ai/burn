use crate::include_models;
include_models!(round);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    fn round_test() {
        // Test for round
        let device = Default::default();
        let model = round::Model::<Backend>::new(&device);

        let input = Tensor::<Backend, 1>::from_floats([-0.5, 1.5, 2.1], &device);
        let expected = Tensor::<Backend, 1>::from_floats([0., 2., 2.], &device);

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
