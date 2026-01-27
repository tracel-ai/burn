// Import the shared macro
use crate::include_models;
include_models!(argmax, argmax_both_keepdims, argmax_1d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn argmax() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmax::Model<TestBackend> = argmax::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[2i64], [2]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn argmax_both_keepdims() {
        // Test both keepdims=True and keepdims=False in the same model
        let model: argmax_both_keepdims::Model<TestBackend> =
            argmax_both_keepdims::Model::default();

        let device = Default::default();
        // Input: [[1.0, 3.0, 2.0], [4.0, 2.0, 1.0]]
        // ArgMax along dim=1 should return:
        // - keepdims=True: [[1], [0]] (indices in 2D format)
        // - keepdims=False: [1, 0] (indices in 1D format)
        let input =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 3.0, 2.0], [4.0, 2.0, 1.0]], &device);
        let (output_keepdims_true, output_keepdims_false) = model.forward(input);

        // Expected outputs based on PyTorch verification:
        // keepdims=True: [[1], [0]] -> shape [2, 1]
        let expected_true = TensorData::from([[1i64], [0]]);
        // keepdims=False: [1, 0] -> shape [2]
        let expected_false = TensorData::from([1i64, 0]);

        output_keepdims_true
            .to_data()
            .assert_eq(&expected_true, true);
        output_keepdims_false
            .to_data()
            .assert_eq(&expected_false, true);
    }

    #[test]
    fn argmax_1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmax_1d::Model<TestBackend> = argmax_1d::Model::default();

        let device = Default::default();
        // Run the model with test input [1.0, 3.0, 2.0, 5.0, 4.0]
        // Expected output: 3 (index of max value 5.0)
        let input = Tensor::<TestBackend, 1>::from_floats([1.0, 3.0, 2.0, 5.0, 4.0], &device);
        let output = model.forward(input);

        // Output should be scalar value 3
        assert_eq!(output, 3i64);
    }
}
