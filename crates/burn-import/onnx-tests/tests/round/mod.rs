use crate::include_models;
include_models!(round);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn round_test() {
        // Test for round
        let device = Default::default();
        let model = round::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 1>::from_floats([-0.5, 1.5, 2.1], &device);
        let expected = Tensor::<TestBackend, 1>::from_floats([0., 2., 2.], &device);

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
