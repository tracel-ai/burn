use crate::include_models;
include_models!(floor);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn floor_test() {
        // Test for floor
        let device = Default::default();
        let model = floor::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 1>::from_floats([-0.5, 1.5, 2.1], &device);
        let expected = Tensor::<TestBackend, 1>::from_floats([-1., 1., 2.], &device);

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
