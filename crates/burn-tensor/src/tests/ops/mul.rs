#[burn_tensor_testgen::testgen(mul)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Int, Tensor, TensorData};

    #[test]
    fn should_support_mul_ops() {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensor::<2>::from_data(data_2, &device);

        let output = tensor_1 * tensor_2;
        let expected = TensorData::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mul_broadcast() {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0]]);
        let data_2 = TensorData::from([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensor::<2>::from_data(data_2, &device);

        let output = tensor_1 * tensor_2;
        let expected = TensorData::from([[0.0, 4.0, 10.0], [0.0, 7.0, 16.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mul_broadcast_2_dims() {
        let device = Default::default();
        let tensor_1 = TestTensor::<1>::from_data([0.0, 1.0, 2.0], &device).reshape([3, 1]);
        let tensor_2 = TestTensor::<1>::from_data([3.0, 4.0, 5.0], &device).reshape([1, 3]);

        let output = tensor_1 * tensor_2;
        let expected = TensorData::from([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_mul_scalar_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let scalar = 2.0;
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor * scalar;
        let expected = TensorData::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_mul_ops_int() {
        let data_1 = TensorData::from([[0, 1, 2], [3, 4, 5]]);
        let data_2 = TensorData::from([[0, 1, 2], [3, 4, 5]]);
        let device = Default::default();
        let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

        let output = tensor_1 * tensor_2;
        let expected = TensorData::from([[0, 1, 4], [9, 16, 25]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mul_broadcast_int() {
        let data_1 = TensorData::from([[0, 1, 2]]);
        let data_2 = TensorData::from([[3, 4, 5], [6, 7, 8]]);
        let device = Default::default();
        let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

        let output = tensor_1 * tensor_2;
        let expected = TensorData::from([[0, 4, 10], [0, 7, 16]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_mul_scalar_ops_int() {
        let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
        let scalar = 2;
        let tensor = TestTensorInt::<2>::from_data(data, &Default::default());

        let output = tensor * scalar;
        let expected = TensorData::from([[0, 2, 4], [6, 8, 10]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_mul_fused() {
        let device = Default::default();
        let tensor1 = TestTensorInt::arange(0..32, &device);
        let tensor2 = TestTensorInt::arange(0..16, &device);
        let tensor3 = TestTensorInt::arange(0..8, &device);

        // Sync prevent fusion.
        let tensor1 = tensor1.reshape([2, 4, 4]);
        TestBackend::sync(&device);
        let tensor2 = tensor2.reshape([1, 4, 4]);
        TestBackend::sync(&device);
        let tensor3 = tensor3.reshape([4, 1, 2]);
        TestBackend::sync(&device);
        let tensor3 = tensor3.swap_dims(0, 2);
        TestBackend::sync(&device);

        let out = tensor1 + tensor2 + tensor3;

        let expected = TensorData::from([
            [
                [0, 4, 8, 12],
                [8, 12, 16, 20],
                [16, 20, 24, 28],
                [24, 28, 32, 36],
            ],
            [
                [17, 21, 25, 29],
                [25, 29, 33, 37],
                [33, 37, 41, 45],
                [41, 45, 49, 53],
            ],
        ]);
        out.to_data().assert_eq(&expected, false);
        panic!("{out}");
    }
}
