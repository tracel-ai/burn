// Import the shared macro
use crate::include_models;
include_models!(argmin, argmin_both_keepdims, argmin_1d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn argmin() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmin::Model<TestBackend> = argmin::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[1.6124, 1.0463, -1.3808], [-0.3852, 0.1301, 0.9780]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[2i64], [0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn argmin_both_keepdims() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmin_both_keepdims::Model<TestBackend> =
            argmin_both_keepdims::Model::default();

        let device = Default::default();
        // Run the model with test input: [[3.0, 1.0, 2.0], [2.0, 4.0, 1.0]]
        let input =
            Tensor::<TestBackend, 2>::from_floats([[3.0, 1.0, 2.0], [2.0, 4.0, 1.0]], &device);
        let (output_keepdims_true, output_keepdims_false) = model.forward(input);

        // Expected: argmin along dim=1
        // For [3.0, 1.0, 2.0]: index of min is 1 (value 1.0)
        // For [2.0, 4.0, 1.0]: index of min is 2 (value 1.0)
        let expected_keepdims_true = TensorData::from([[1i64], [2]]); // Shape [2, 1]
        let expected_keepdims_false = TensorData::from([1i64, 2]); // Shape [2]

        output_keepdims_true
            .to_data()
            .assert_eq(&expected_keepdims_true, true);
        output_keepdims_false
            .to_data()
            .assert_eq(&expected_keepdims_false, true);
    }

    #[test]
    fn argmin_1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmin_1d::Model<TestBackend> = argmin_1d::Model::default();

        let device = Default::default();
        // Run the model with test input [5.0, 3.0, 2.0, 1.0, 4.0]
        // Expected output: 3 (index of min value 1.0)
        let input = Tensor::<TestBackend, 1>::from_floats([5.0, 3.0, 2.0, 1.0, 4.0], &device);
        let output = model.forward(input);

        // Output should be scalar value 3
        assert_eq!(output, 3i64);
    }
}
