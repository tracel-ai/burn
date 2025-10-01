// Import the shared macro
use crate::include_models;
include_models!(sinh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn sinh() {
        let device = Default::default();
        let model: sinh::Model<TestBackend> = sinh::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[-4.0, 0.5, 1.0, 9.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-27.2899, 0.5211, 1.1752, 4051.5419]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
