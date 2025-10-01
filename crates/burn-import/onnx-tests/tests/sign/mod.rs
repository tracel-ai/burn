// Import the shared macro
use crate::include_models;
include_models!(sign);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn sign() {
        let device = Default::default();
        let model: sign::Model<TestBackend> = sign::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[-1.0, 2.0, 0.0, -4.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-1.0f32, 1.0, 0.0, -1.0]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
