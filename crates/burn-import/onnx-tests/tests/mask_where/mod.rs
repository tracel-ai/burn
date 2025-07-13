// Import the shared macro
use crate::include_models;
include_models!(
    mask_where,
    mask_where_broadcast,
    mask_where_scalar_x,
    mask_where_scalar_y,
    mask_where_all_scalar
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn mask_where() {
        let device = Default::default();
        let model: mask_where::Model<Backend> = mask_where::Model::new(&device);

        let x = Tensor::ones([2, 2], &device);
        let y = Tensor::zeros([2, 2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mask_where_broadcast() {
        let device = Default::default();
        let model: mask_where_broadcast::Model<Backend> = mask_where_broadcast::Model::new(&device);

        let x = Tensor::ones([2], &device);
        let y = Tensor::zeros([2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mask_where_scalar_x() {
        let device = Default::default();
        let model: mask_where_scalar_x::Model<Backend> = mask_where_scalar_x::Model::new(&device);

        let x = 1.0f32;
        let y = Tensor::zeros([2, 2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mask_where_scalar_y() {
        let device = Default::default();
        let model: mask_where_scalar_y::Model<Backend> = mask_where_scalar_y::Model::new(&device);

        let x = Tensor::ones([2, 2], &device);
        let y = 0.0f32;
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mask_where_all_scalar() {
        let device = Default::default();
        let model: mask_where_all_scalar::Model<Backend> =
            mask_where_all_scalar::Model::new(&device);

        let x = 1.0f32;
        let y = 0.0f32;
        let mask = true;

        let output = model.forward(mask, x, y);
        let expected = 1.0f32;

        assert_eq!(output, expected);
    }
}
