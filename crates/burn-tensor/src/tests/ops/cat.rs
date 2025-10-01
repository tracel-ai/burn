#[burn_tensor_testgen::testgen(cat)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Bool, DType, Int, Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_cat_ops_2d_dim0() {
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]], &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_cat_ops_int() {
        let device = Default::default();
        let tensor_1 = TestTensorInt::<2>::from_data([[1, 2, 3]], &device);
        let tensor_2 = TestTensorInt::<2>::from_data([[4, 5, 6]], &device);

        let output = Tensor::cat(vec![tensor_1, tensor_2], 0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1, 2, 3], [4, 5, 6]]), false);
    }

    #[test]
    fn should_support_cat_ops_bool() {
        let device = Default::default();
        let tensor_1 = TestTensorBool::<2>::from_data([[false, true, true]], &device);
        let tensor_2 = TestTensorBool::<2>::from_data([[true, true, false]], &device);

        let output = Tensor::cat(vec![tensor_1, tensor_2], 0);

        output.into_data().assert_eq(
            &TensorData::from([[false, true, true], [true, true, false]]),
            false,
        );
    }

    #[test]
    fn should_support_cat_ops_2d_dim1() {
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]], &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 1);
        let expected = TensorData::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_cat_ops_3d() {
        let device = Default::default();
        let tensor_1 = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
        let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]], &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0]]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0]], &device);

        TestTensor::cat(vec![tensor_1, tensor_2], 0).into_data();
    }

    #[test]
    #[should_panic]
    fn should_panic_when_list_of_vectors_is_empty() {
        let tensor: Vec<TestTensor<2>> = vec![];
        TestTensor::cat(tensor, 0).into_data();
    }

    #[test]
    #[should_panic]
    fn should_panic_when_cat_exceeds_dimension() {
        let device = Default::default();
        let tensor_1 = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
        let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]], &device);

        TestTensor::cat(vec![tensor_1, tensor_2], 3).into_data();
    }

    #[test]
    fn should_support_cat_ops_cast_dtype() {
        let device = Default::default();
        // ok for f32 backends, casts dtype for f16 tests
        let tensor_1 = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device)
            .cast(DType::F32);
        let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]], &device).cast(DType::F32);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0]]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
