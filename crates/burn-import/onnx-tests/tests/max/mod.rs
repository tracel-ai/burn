// Import the shared macro
use crate::include_models;
include_models!(max);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn max() {
        let device = Default::default();

        let model: max::Model<TestBackend> = max::Model::new(&device);
        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 42.0, 9.0, 42.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[42.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[42.0f32, 42.0, 42.0, 42.0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
