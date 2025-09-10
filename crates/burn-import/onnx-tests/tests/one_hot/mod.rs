use crate::include_models;
include_models!(one_hot);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn one_hot() {
        // Test for OneHot model
        let device = Default::default();
        let model = one_hot::Model::<TestBackend>::new(&device);
        let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints([1, 0, 2], &device);
        let expected: Tensor<TestBackend, 2, burn::prelude::Float> =
            Tensor::from_data(TensorData::from([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), &device);
        let output: Tensor<TestBackend, 2, Int> = model.forward(input);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
