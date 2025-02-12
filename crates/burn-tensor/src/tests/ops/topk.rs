#[burn_tensor_testgen::testgen(topk)]
mod tests {
    use super::*;
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_topk_1d() {
        // Int
        let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

        let values = tensor.topk(3, /*dim*/ 0);
        let expected = TensorData::from([5, 4, 3]);

        values.into_data().assert_eq(&expected, false);

        // Float
        let tensor = TestTensor::<1>::from([1., 2., 3., 4., 5.]);

        let values = tensor.topk(3, /*dim*/ 0);
        let expected = TensorData::from([5., 4., 3.]);

        values.into_data().assert_approx_eq(&expected, 5);
    }

    #[test]
    fn test_topk() {
        // 3D Int
        let tensor = TestTensorInt::<3>::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        let values = tensor.topk(2, /*dim*/ 2);
        let expected = TensorData::from([[[7, 4], [6, 5]], [[9, 3], [8, 8]]]);

        values.into_data().assert_eq(&expected, false);

        // 3D Float
        let tensor =
            TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 8.]]]);

        let values = tensor.topk(2, /*dim*/ 2);
        let expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 8.]]]);

        values.into_data().assert_approx_eq(&expected, 5);
    }

    #[test]
    fn test_topk_with_indices_1d() {
        let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

        let (values, indices) = tensor.topk_with_indices(3, /*dim*/ 0);

        let values_expected = TensorData::from([5, 4, 3]);
        values.into_data().assert_eq(&values_expected, false);

        let indices_expected = TensorData::from([4, 3, 2]);
        indices.into_data().assert_eq(&indices_expected, false);
    }

    #[test]
    fn test_topk_with_indices_3d() {
        let tensor =
            TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

        let (values, indices) = tensor.topk_with_indices(2, /*dim*/ 2);

        let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

        values.into_data().assert_approx_eq(&values_expected, 5);

        let indices_expected = TensorData::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);

        indices.into_data().assert_eq(&indices_expected, false);
    }
}
