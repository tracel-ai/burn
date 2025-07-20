// Import the shared macro
use crate::include_models;
include_models!(or);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn or() {
        let device = Default::default();
        let model: or::Model<Backend> = or::Model::new(&device);

        let input_x = Tensor::<Backend, 4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<Backend, 4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, true, true, true]]]]);

        output.assert_eq(&expected, true);
    }
}
