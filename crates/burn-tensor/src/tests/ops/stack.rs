#[burn_tensor_testgen::testgen(stack)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_stack_ops_2d_dim0() {
        let device = Default::default();
        let tensor_1: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 2.0, 3.0]], &device);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([[4.0, 5.0, 6.0]], &device);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_stack_ops_int() {
        let device = Default::default();
        let tensor_1 = TestTensorInt::<2>::from_data([[1, 2, 3]], &device);
        let tensor_2 = TestTensorInt::<2>::from_data([[4, 5, 6]], &device);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1, 2, 3]], [[4, 5, 6]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_stack_ops_bool() {
        let device = Default::default();
        let tensor_1 = TestTensorBool::<2>::from_data([[false, true, true]], &device);
        let tensor_2 = TestTensorBool::<2>::from_data([[true, true, false]], &device);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[false, true, true]], [[true, true, false]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_stack_ops_2d_dim1() {
        let device = Default::default();
        let tensor_1: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 2.0, 3.0]], &device);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([[4.0, 5.0, 6.0]], &device);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 1);
        let expected = TensorData::from([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_stack_ops_3d() {
        let device = Default::default();
        let tensor_1: Tensor<TestBackend, 3> =
            TestTensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
        let tensor_2: Tensor<TestBackend, 3> =
            TestTensor::from_data([[[4.0, 5.0, 6.0]], [[4.1, 5.1, 6.1]]], &device);

        let output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([
            [[[1.0000, 2.0000, 3.0000]], [[1.1000, 2.1000, 3.1000]]],
            [[[4.0000, 5.0000, 6.0000]], [[4.1000, 5.1000, 6.1000]]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let device = Default::default();
        let tensor_1: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([[4.0, 5.0]], &device);

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
        let device = Default::default();
        let tensor_1: Tensor<TestBackend, 3> =
            Tensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
        let tensor_2: Tensor<TestBackend, 3> = Tensor::from_data([[[4.0, 5.0, 6.0]]], &device);

        let output: Tensor<TestBackend, 4> = TestTensor::stack(vec![tensor_1, tensor_2], 3);
    }

    #[test]
    fn should_generate_row_major_layout() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(1..25, &device).reshape([4, 6]);
        let zeros: Tensor<TestBackend, 2, Int> = Tensor::zeros([4, 6], &device);
        let intersperse =
            Tensor::stack::<3>([tensor.clone(), zeros.clone()].to_vec(), 2).reshape([4, 12]);

        let expected = TensorData::from([
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0],
            [7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0],
            [13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0],
            [19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0],
        ]);

        intersperse.into_data().assert_eq(&expected, false);
    }
}
