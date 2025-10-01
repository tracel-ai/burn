// Import the shared macro
use crate::include_models;
include_models!(or, or_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn or() {
        let device = Default::default();
        let model: or::Model<TestBackend> = or::Model::new(&device);

        let input_x = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, true, true, true]]]]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn or_scalar() {
        let device = Default::default();
        let model: or_scalar::Model<TestBackend> = or_scalar::Model::new(&device);

        // Test various combinations of scalar boolean inputs
        // (input1 || true) || (input2 || false) = true || input2 = true
        assert_eq!(model.forward(false, false), true);
        assert_eq!(model.forward(false, true), true);
        assert_eq!(model.forward(true, false), true);
        assert_eq!(model.forward(true, true), true);
    }
}
