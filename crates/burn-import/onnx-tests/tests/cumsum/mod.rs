// Include the models for this node type
use crate::include_models;
include_models!(
    cumsum,
    cumsum_exclusive,
    cumsum_reverse,
    cumsum_exclusive_reverse,
    cumsum_2d,
    cumsum_runtime_axis,
    cumsum_single_element,
    cumsum_exclusive_single
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn cumsum_default() {
        // Input: [1., 2., 3., 4., 5.]
        // Expected output: [1., 3., 6., 10., 15.]
        let device = Default::default();
        let model: cumsum::Model<TestBackend> = cumsum::Model::default();

        let input = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([1f32, 3., 6., 10., 15.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_exclusive_mode() {
        // Input: [1., 2., 3., 4., 5.]
        // Expected output (exclusive): [0., 1., 3., 6., 10.]
        let device = Default::default();
        let model: cumsum_exclusive::Model<TestBackend> = cumsum_exclusive::Model::default();

        let input = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([0f32, 1., 3., 6., 10.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_reverse_mode() {
        // Input: [1., 2., 3., 4., 5.]
        // Expected output (reverse): [15., 14., 12., 9., 5.]
        let device = Default::default();
        let model: cumsum_reverse::Model<TestBackend> = cumsum_reverse::Model::default();

        let input = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([15f32, 14., 12., 9., 5.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_exclusive_reverse_mode() {
        // Input: [1., 2., 3., 4., 5.]
        // Expected output (exclusive+reverse): [14., 12., 9., 5., 0.]
        let device = Default::default();
        let model: cumsum_exclusive_reverse::Model<TestBackend> =
            cumsum_exclusive_reverse::Model::default();

        let input = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([14f32, 12., 9., 5., 0.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_2d_axis_1() {
        // Input: [[1., 2., 3.], [4., 5., 6.]]
        // Expected output (2D, axis=1): [[1., 3., 6.], [4., 9., 15.]]
        let device = Default::default();
        let model: cumsum_2d::Model<TestBackend> = cumsum_2d::Model::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[1f32, 3., 6.], [4., 9., 15.]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_runtime_axis_test() {
        // Input: [[1., 2., 3.], [4., 5., 6.]]
        // Axis: 1 (runtime input as scalar)
        // Expected output (2D, axis=1): [[1., 3., 6.], [4., 9., 15.]]
        let device = Default::default();
        let model: cumsum_runtime_axis::Model<TestBackend> = cumsum_runtime_axis::Model::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let axis: i64 = 1; // Runtime axis value (scalar)
        let output = model.forward(input, axis);
        let expected = TensorData::from([[1f32, 3., 6.], [4., 9., 15.]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_single_element_test() {
        // Edge case: single element tensor
        // Input: [42.]
        // Expected output: [42.]
        let device = Default::default();
        let model: cumsum_single_element::Model<TestBackend> =
            cumsum_single_element::Model::default();

        let input = Tensor::<TestBackend, 1>::from_floats([42.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([42f32]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn cumsum_exclusive_single_element_test() {
        // Edge case: exclusive cumsum on single element
        // Tests dim_size - 1 = 0 case
        // Input: [42.]
        // Expected output (exclusive): [0.] (sum of nothing before first element)
        let device = Default::default();
        let model: cumsum_exclusive_single::Model<TestBackend> =
            cumsum_exclusive_single::Model::default();

        let input = Tensor::<TestBackend, 1>::from_floats([42.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([0f32]);

        output.to_data().assert_eq(&expected, true);
    }
}
