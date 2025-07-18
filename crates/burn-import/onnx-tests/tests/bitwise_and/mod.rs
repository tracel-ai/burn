// Include the models for this node type
use crate::include_models;
include_models!(bitwise_and, bitwise_and_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn bitwise_and_tensors() {
        let device = Default::default();
        let model: bitwise_and::Model<Backend> = bitwise_and::Model::new(&device);
        // Run the model
        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<Backend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([1i64, 0, 2, 0]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitwise_and_scalar_tensor() {
        let device = Default::default();
        let model: bitwise_and_scalar::Model<Backend> = bitwise_and_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([0i64, 2, 2, 0]);
        output.to_data().assert_eq(&expected, true);
    }
}
