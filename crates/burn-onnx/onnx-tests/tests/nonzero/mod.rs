use crate::include_models;
include_models!(
    nonzero_float32,
    nonzero_int64,
    nonzero_bool,
    nonzero_1d,
    nonzero_3d,
    nonzero_empty
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Int, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn nonzero_float32_test() {
        let device = Default::default();
        let model = nonzero_float32::Model::<TestBackend>::new(&device);

        // Create a 3x4 tensor with some non-zero values
        // Expected nonzero indices: (0,1), (1,2), (2,0), (2,3)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0],
                [3.1, 0.0, 0.0, -1.2],
            ],
            &device,
        );

        let output = model.forward(input);

        // Expected indices in ONNX format [rank, num_nonzero]: [[0,1,2,2], [1,2,0,3]]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints(
            [
                [0, 1, 2, 2], // Row indices of non-zero elements
                [1, 2, 0, 3], // Column indices of non-zero elements
            ],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn nonzero_int64_test() {
        let device = Default::default();
        let model = nonzero_int64::Model::<TestBackend>::new(&device);

        // Create a 2x3 int tensor with some non-zero values
        // Expected nonzero indices: (0,0), (1,2)
        let input = Tensor::<TestBackend, 2, Int>::from_ints([[5, 0, 0], [0, 0, -3]], &device);

        let output = model.forward(input);

        // Expected indices in ONNX format [rank, num_nonzero]: [[0,1], [0,2]]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints(
            [
                [0, 1], // Row indices of non-zero elements
                [0, 2], // Column indices of non-zero elements
            ],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn nonzero_bool_test() {
        let device = Default::default();
        let model = nonzero_bool::Model::<TestBackend>::new(&device);

        // Create a 2x2 bool tensor
        // Expected nonzero indices: (0,1), (1,0)
        let input = Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, true], [true, false]].into(),
            &device,
        );

        let output = model.forward(input);

        // Expected indices in ONNX format [rank, num_nonzero]: [[0,1], [1,0]]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints(
            [
                [0, 1], // Row indices of non-zero elements
                [1, 0], // Column indices of non-zero elements
            ],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn nonzero_1d_test() {
        let device = Default::default();
        let model = nonzero_1d::Model::<TestBackend>::new(&device);

        // Create a 1D tensor with some non-zero values
        // Expected nonzero indices: 1, 3, 5
        let input = Tensor::<TestBackend, 1>::from_floats([0.0, 2.0, 0.0, -1.0, 0.0, 3.5], &device);

        let output = model.forward(input);

        // Expected indices in ONNX format [rank, num_nonzero]: [[1, 3, 5]]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints([[1, 3, 5]], &device);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn nonzero_3d_test() {
        let device = Default::default();
        let model = nonzero_3d::Model::<TestBackend>::new(&device);

        // Create a 2x2x3 tensor with a few non-zero values
        // Expected nonzero indices: (0,0,1), (1,1,2)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            ],
            &device,
        );

        let output = model.forward(input);

        // Expected indices in ONNX format [rank, num_nonzero]: [[0,1], [0,1], [1,2]]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints(
            [
                [0, 1], // Dimension 0 indices of non-zero elements
                [0, 1], // Dimension 1 indices of non-zero elements
                [1, 2], // Dimension 2 indices of non-zero elements
            ],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn nonzero_empty_test() {
        let device = Default::default();
        let model = nonzero_empty::Model::<TestBackend>::new(&device);

        // Create a tensor with all zeros
        let input =
            Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);

        let output = model.forward(input);

        // Empty tensor: [2, 0] - 2 dimensions, 0 non-zero elements
        let expected = Tensor::<TestBackend, 2, Int>::empty([2, 0], &device);

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
