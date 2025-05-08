use crate::include_models;
include_models!(tile);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn tile() {
        let device = Default::default();
        let model: tile::Model<Backend> = tile::Model::new(&device);

        let input = Tensor::<Backend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [1.0f32, 2.0f32, 1.0f32, 2.0f32],
            [3.0f32, 4.0f32, 3.0f32, 4.0f32],
            [1.0f32, 2.0f32, 1.0f32, 2.0f32],
            [3.0f32, 4.0f32, 3.0f32, 4.0f32],
        ]);

        output.assert_eq(&expected, true);
    }
}
