// Import the shared macro
use crate::include_models;
include_models!(min);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn min() {
        let device = Default::default();

        let model: min::Model<TestBackend> = min::Model::new(&device);
        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0, 42.0, 0.0, 42.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[2.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[-1.0f32, 4.0, 0.0, 25.0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
