// Import the shared macro
use crate::include_models;
include_models!(xor, xor_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn xor() {
        let device = Default::default();
        let model: xor::Model<TestBackend> = xor::Model::new(&device);

        let input_x = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, true, true, false]]]]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn xor_scalar() {
        let device = Default::default();
        let model: xor_scalar::Model<TestBackend> = xor_scalar::Model::new(&device);

        // Test various combinations of scalar boolean inputs
        // (input1 ^ true) ^ (input2 ^ false) = (!input1) ^ input2
        assert_eq!(model.forward(false, false), true); // true ^ false = true
        assert_eq!(model.forward(false, true), false); // true ^ true = false
        assert_eq!(model.forward(true, false), false); // false ^ false = false
        assert_eq!(model.forward(true, true), true); // false ^ true = true
    }
}
