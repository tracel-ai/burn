#[burn_tensor_testgen::testgen(topk)]
mod tests {
    use super::*;
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_topk_1d() {
        // Int
        // largest
        let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

        let values = tensor.topk(3, /*dim*/ 0, /*largest*/ Some(1));
        let expected = TensorData::from([5, 4, 3]);

        values.into_data().assert_eq(&expected, false);

        // smallest
        let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

        let values = tensor.topk(3, /*dim*/ 0, /*largest*/ Some(0));
        let expected = TensorData::from([1, 2, 3]);

        values.into_data().assert_eq(&expected, false);

        // Float
        // largest
        let tensor = TestTensor::<1>::from([1., 2., 3., 4., 5.]);

        let values = tensor.topk(3, /*dim*/ 0, /*largest*/ Some(1));
        let expected = TensorData::from([5., 4., 3.]);

        values.into_data().assert_approx_eq(&expected, 5);
        // Float
        // smallest
        let tensor = TestTensor::<1>::from([1., 2., 3., 4., 5.]);

        let values = tensor.topk(3, /*dim*/ 0, /*largest*/ Some(0));
        let expected = TensorData::from([1., 2., 3.]);

        values.into_data().assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_topk() {
        // 3D Int
        // largest
        let tensor = TestTensorInt::<3>::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        let values = tensor.topk(2, /*dim*/ 2, /*largest*/ Some(1));
        let expected = TensorData::from([[[7, 4], [6, 5]], [[9, 3], [8, 8]]]);

        values.into_data().assert_eq(&expected, false);

        // smallest
        let tensor = TestTensorInt::<3>::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        let values = tensor.topk(2, /*dim*/ 2, /*largest*/ Some(0));
        let expected = TensorData::from([[[1, 4], [2, 5]], [[0, 3], [2, 8]]]);

        values.into_data().assert_eq(&expected, false);

        // 3D Float
        // largest
        let tensor =
            TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 8.]]]);

        let values = tensor.topk(2, /*dim*/ 2, /*largest*/ Some(1));
        let expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 8.]]]);

        values.into_data().assert_approx_eq(&expected, 5);

        // smallest
        let tensor =
            TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 8.]]]);

        let values = tensor.topk(2, /*dim*/ 2, /*largest*/ Some(0));
        let expected = TensorData::from([[[1, 4], [2, 5]], [[0, 3], [2, 8]]]);
    }

    #[test]
    fn test_topk_with_indices() {
        // 1D
        // largest
        let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

        let (values, indices) =
            tensor.topk_with_indices(3, /*dim*/ 0, /*largest*/ Some(1));

        let values_expected = TensorData::from([5, 4, 3]);
        values.into_data().assert_eq(&values_expected, false);

        let indices_expected = TensorData::from([4, 3, 2]);
        indices.into_data().assert_eq(&indices_expected, false);

        // smallest
        let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

        let (values, indices) =
            tensor.topk_with_indices(3, /*dim*/ 0, /*largest*/ Some(0));

        let values_expected = TensorData::from([1, 2, 3]);
        values.into_data().assert_eq(&values_expected, false);

        let indices_expected = TensorData::from([0, 1, 2]);
        indices.into_data().assert_eq(&indices_expected, false);

        // 3D
        // largest
        let tensor =
            TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

        let (values, indices) =
            tensor.topk_with_indices(2, /*dim*/ 2, /*largest*/ Some(1));

        let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

        values.into_data().assert_approx_eq(&values_expected, 5);

        let indices_expected = TensorData::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);

        indices.into_data().assert_eq(&indices_expected, false);

        // smallest
        let tensor =
            TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

        let (values, indices) =
            tensor.topk_with_indices(2, /*dim*/ 2, /*largest*/ Some(0));

        let values_expected = TensorData::from([[[1., 4.], [2., 5.]], [[0., 3.], [2., 7.]]]);

        values.into_data().assert_approx_eq(&values_expected, 5);

        let indices_expected = TensorData::from([[[0, 1], [0, 1]], [[1, 0], [1, 2]]]);

        indices.into_data().assert_eq(&indices_expected, false);
    }
}
