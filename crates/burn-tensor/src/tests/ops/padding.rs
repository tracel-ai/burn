#[burn_tensor_testgen::testgen(padding)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Int, Numeric, Shape, Tensor, TensorData};

    #[test]
    fn padding_2d_test() {
        let unpadded_floats: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let tensor = TestTensor::<2>::from(unpadded_floats);

        let padded_tensor = tensor.pad((2, 2, 2, 2), 1.1);

        let expected = TensorData::from([
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 2.0, 1.1, 1.1],
            [1.1, 1.1, 3.0, 4.0, 5.0, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        ]);
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_4d_test() {
        let unpadded_floats = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]];
        let tensor = TestTensor::<4>::from(unpadded_floats);

        let padded_tensor = tensor.pad((2, 2, 2, 2), 1.1);

        let expected = TensorData::from([[[
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 1.1, 1.1],
            [1.1, 1.1, 2.0, 3.0, 1.1, 1.1],
            [1.1, 1.1, 4.0, 5.0, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        ]]]);
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_asymmetric_test() {
        let unpadded_floats = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]];
        let tensor = TestTensor::<4>::from(unpadded_floats);

        let padded_tensor = tensor.pad((2, 1, 4, 3), 1.1);

        let expected = TensorData::from([[[
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 1.1],
            [1.1, 1.1, 2.0, 3.0, 1.1],
            [1.1, 1.1, 4.0, 5.0, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
        ]]]);
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_asymmetric_integer_test() {
        let unpadded_ints = [[[[0, 1], [2, 3], [4, 5]]]];

        let tensor = TestTensorInt::<4>::from(unpadded_ints);
        let padded_tensor = tensor.pad((2, 1, 4, 3), 6);

        let padded_primitive_data_expected = [[[
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 0, 1, 6],
            [6, 6, 2, 3, 6],
            [6, 6, 4, 5, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
        ]]];
        let expected = TensorData::from([[[
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 0, 1, 6],
            [6, 6, 2, 3, 6],
            [6, 6, 4, 5, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6],
        ]]]);
        padded_tensor.into_data().assert_eq(&expected, false);
    }
}
