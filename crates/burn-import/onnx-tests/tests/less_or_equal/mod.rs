// Import the shared macro
use crate::include_models;
include_models!(less_or_equal, less_or_equal_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn less_or_equal() {
        let device = Default::default();
        let model: less_or_equal::Model<TestBackend> = less_or_equal::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, true, false, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_or_equal_scalar() {
        let device = Default::default();
        let model: less_or_equal_scalar::Model<TestBackend> =
            less_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, false, false, true]]);

        output.to_data().assert_eq(&expected, true);
    }
}
