// Import the shared macro
use crate::include_models;
include_models!(greater, greater_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn greater() {
        let device = Default::default();
        let model: greater::Model<TestBackend> = greater::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, false, true, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_scalar() {
        let device = Default::default();
        let model: greater_scalar::Model<TestBackend> = greater_scalar::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, true, true, false]]);

        output.to_data().assert_eq(&expected, true);
    }
}
