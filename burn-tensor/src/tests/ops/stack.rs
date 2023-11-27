#[burn_tensor_testgen::testgen(stack)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn should_support_stack_ops_2d_dim0() {
        let tensor_1: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 2.0, 3.0]]);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([[4.0, 5.0, 6.0]]);

        let output = Tensor::stack(vec![tensor_1, tensor_2], 0);

        let data_expected = Data::from([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]);
        output.into_data().assert_approx_eq(&data_expected, 3);
    }

    #[test]
    fn should_support_stack_ops_int() {
        let tensor_1 = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3]]);
        let tensor_2 = Tensor::<TestBackend, 2, Int>::from_data([[4, 5, 6]]);

        let output = Tensor::stack(vec![tensor_1, tensor_2], 0);

        let data_expected = Data::from([[[1, 2, 3]], [[4, 5, 6]]]);
        assert_eq!(&output.into_data(), &data_expected);
    }

    #[test]
    fn should_support_stack_ops_bool() {
        let tensor_1 = Tensor::<TestBackend, 2, Bool>::from_data([[false, true, true]]);
        let tensor_2 = Tensor::<TestBackend, 2, Bool>::from_data([[true, true, false]]);

        let output = Tensor::stack(vec![tensor_1, tensor_2], 0);

        let data_expected = Data::from([[[false, true, true]], [[true, true, false]]]);
        assert_eq!(&output.into_data(), &data_expected);
    }

    #[test]
    fn should_support_stack_ops_2d_dim1() {
        let tensor_1: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 2.0, 3.0]]);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([[4.0, 5.0, 6.0]]);

        let output = Tensor::stack(vec![tensor_1, tensor_2], 1);

        let data_expected = Data::from([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
        output.into_data().assert_approx_eq(&data_expected, 3);
    }

    #[test]
    fn should_support_stack_ops_3d() {
        let tensor_1: Tensor<TestBackend, 3> =
            TestTensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]]);
        let tensor_2: Tensor<TestBackend, 3> =
            TestTensor::from_data([[[4.0, 5.0, 6.0]], [[4.1, 5.1, 6.1]]]);

        let output = Tensor::stack(vec![tensor_1, tensor_2], 0);

        let data_expected = Data::from([
            [[[1.0000, 2.0000, 3.0000]], [[1.1000, 2.1000, 3.1000]]],
            [[[4.0000, 5.0000, 6.0000]], [[4.1000, 5.1000, 6.1000]]],
        ]);
        output.into_data().assert_approx_eq(&data_expected, 3);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let tensor_1: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([[4.0, 5.0]]);

        let output: Tensor<TestBackend, 3> = Tensor::stack(vec![tensor_1, tensor_2], 0);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_list_of_vectors_is_empty() {
        let tensors: Vec<Tensor<TestBackend, 2>> = vec![];
        let output: Tensor<TestBackend, 3> = TestTensor::stack(tensors, 0);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_stack_exceeds_dimension() {
        let tensor_1: Tensor<TestBackend, 3> =
            Tensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]]);
        let tensor_2: Tensor<TestBackend, 3> = Tensor::from_data([[[4.0, 5.0, 6.0]]]);

        let output: Tensor<TestBackend, 4> = TestTensor::stack(vec![tensor_1, tensor_2], 3);
    }
}
