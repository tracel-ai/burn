// Import the shared macro
use crate::include_models;
include_models!(abs);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn abs() {
        let device = Default::default();
        let model: abs::Model<TestBackend> = abs::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[-1.0, -4.0, 9.0, -25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1.0f32, 4.0, 9.0, 25.0]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
