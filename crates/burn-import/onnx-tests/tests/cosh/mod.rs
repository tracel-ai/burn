// Import the shared macro
use crate::include_models;
include_models!(cosh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn cosh() {
        let device = Default::default();
        let model: cosh::Model<TestBackend> = cosh::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[-4.0, 0.5, 1.0, 9.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[27.3082, 1.1276, 1.5431, 4051.5420]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
