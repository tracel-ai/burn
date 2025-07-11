// Include the models for this node type
use crate::include_models;
include_models!(bitshift_left, bitshift_left_scalar, bitshift_right, bitshift_right_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn bitshift_left_tensors() {
        // Initialize the model with weights (loaded from the exported file)
        let device = Default::default();
        let model: bitshift_left::Model<Backend> = bitshift_left::Model::new(&device);
        // Run the model
        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<Backend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([2i64, 4, 12, 16]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitshift_left_scalar_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let device = Default::default();
        let model: bitshift_left_scalar::Model<Backend> = bitshift_left_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([4i64, 8, 12, 16]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitshift_right_tensors() {
        let device = Default::default();
        let model: bitshift_right::Model<Backend> = bitshift_right::Model::new(&device);

        // Run the model
        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<Backend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([0i64, 1, 0, 1]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitshift_right_scalar_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let device = Default::default();
        let model: bitshift_right_scalar::Model<Backend> =
            bitshift_right_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([0i64, 0, 0, 1]);

        output.to_data().assert_eq(&expected, true);
    }
}