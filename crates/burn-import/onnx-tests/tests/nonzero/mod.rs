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

        // Expected indices in row-major order: [0,1], [1,2], [2,0], [2,3]
        let expected =
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1], [1, 2], [2, 0], [2, 3]], &device);

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

        // Expected indices: [0,0], [1,2]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints([[0, 0], [1, 2]], &device);

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

        // Expected indices: [0,1], [1,0]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1], [1, 0]], &device);

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

        // Expected indices: 1, 3, 5
        let expected = Tensor::<TestBackend, 2, Int>::from_ints([[1], [3], [5]], &device);

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

        // Expected indices: [0,0,1], [1,1,2]
        let expected = Tensor::<TestBackend, 2, Int>::from_ints([[0, 0, 1], [1, 1, 2]], &device);

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

        // Empty tensor: [0, 2] - no non-zero elements, but still 2D coordinates
        let expected = Tensor::<TestBackend, 2, Int>::empty([0, 2], &device);

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
