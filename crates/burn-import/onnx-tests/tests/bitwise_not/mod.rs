// Include the models for this node type
use crate::include_models;
include_models!(bitwise_not);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn bitwise_not_tensors() {
        let device = Default::default();
        let model: bitwise_not::Model<Backend> = bitwise_not::Model::new(&device);
        // Run the model
        let input = Tensor::<Backend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[-2i64, -3, -4, -5]]);
        output.to_data().assert_eq(&expected, true);
    }
}