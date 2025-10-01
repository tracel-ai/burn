// Import the shared macro
use crate::include_models;
include_models!(erf);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn erf() {
        let model: erf::Model<TestBackend> = erf::Model::default();

        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_data([[[[1.0, 2.0, 3.0, 4.0]]]], &device);
        let output = model.forward(input);
        let expected =
            Tensor::<TestBackend, 4>::from_data([[[[0.8427f32, 0.9953, 1.0000, 1.0000]]]], &device);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
