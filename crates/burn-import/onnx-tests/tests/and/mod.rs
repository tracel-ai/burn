// Import the shared macro
use crate::include_models;
include_models!(and);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn and() {
        let device = Default::default();
        let model: and::Model<Backend> = and::Model::new(&device);

        let input_x = Tensor::<Backend, 4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<Backend, 4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, false, false, true]]]]);

        output.assert_eq(&expected, true);
    }
}
