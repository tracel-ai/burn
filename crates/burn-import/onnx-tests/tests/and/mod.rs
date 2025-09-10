// Import the shared macro
use crate::include_models;
include_models!(and, and_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn and() {
        let device = Default::default();
        let model: and::Model<TestBackend> = and::Model::new(&device);

        let input_x = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, false, false, true]]]]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn and_scalar() {
        let device = Default::default();
        let model: and_scalar::Model<TestBackend> = and_scalar::Model::new(&device);

        // Test various combinations of scalar boolean inputs
        // (input1 && true) && (input2 && false) = input1 && false = false
        assert_eq!(model.forward(false, false), false);
        assert_eq!(model.forward(false, true), false);
        assert_eq!(model.forward(true, false), false);
        assert_eq!(model.forward(true, true), false); // true && false = false
    }
}
