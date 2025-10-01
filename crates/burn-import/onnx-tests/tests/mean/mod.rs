// Import the shared macro
use crate::include_models;
include_models!(mean);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn mean_tensor_and_tensor() {
        let device = Default::default();
        let model: mean::Model<TestBackend> = mean::Model::default();

        let input1 = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4.], &device);
        let input2 = Tensor::<TestBackend, 1>::from_floats([2., 2., 4., 0.], &device);
        let input3 = Tensor::<TestBackend, 1>::from_floats([3., 2., 5., -4.], &device);

        let output = model.forward(input1, input2, input3);
        let expected = TensorData::from([2.0f32, 2., 4., 0.]);

        output.to_data().assert_eq(&expected, true);
    }
}
