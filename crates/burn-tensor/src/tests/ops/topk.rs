#[burn_tensor_testgen::testgen(topk)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn test_topk_1d() {
        // Int
        let tensor = TestTensorInt::from([1, 2, 3, 4, 5]);

        let values = tensor.topk(3, /*dim*/ 0);
        let values_actual = values.into_data();

        let values_expected = Data::from([5, 4, 3]);
        assert_eq!(values_expected, values_actual);

        // Float
        let tensor = TestTensor::from([1., 2., 3., 4., 5.]);

        let values = tensor.topk(3, /*dim*/ 0);
        let values_actual = values.into_data();

        let values_expected = Data::from([5., 4., 3.]);
        values_expected.assert_approx_eq(&values_actual, 5);
    }

    #[test]
    fn test_topk() {
        // 2D Int
        let tensor = TestTensorInt::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        let values = tensor.topk(2, /*dim*/ 2);
        let values_actual = values.into_data();

        let values_expected = Data::from([[[7, 4], [6, 5]], [[9, 3], [8, 8]]]);
        assert_eq!(values_expected, values_actual);

        // 2D Float
        let tensor = TestTensor::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 8.]]]);

        let values = tensor.topk(2, /*dim*/ 2);
        let values_actual = values.into_data();

        let values_expected = Data::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 8.]]]);
        values_expected.assert_approx_eq(&values_actual, 5);
    }

    #[test]
    fn test_topk_with_indices() {
        // 1D
        let tensor = TestTensorInt::from([1, 2, 3, 4, 5]);

        let (values, indices) = tensor.topk_with_indices(3, /*dim*/ 0);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([5, 4, 3]);
        assert_eq!(values_expected, values_actual);

        let indices_expected = Data::from([4, 3, 2]);
        assert_eq!(indices_expected, indices_actual);

        // 2D
        let tensor = TestTensor::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

        let (values, indices) = tensor.topk_with_indices(2, /*dim*/ 2);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);
        values_expected.assert_approx_eq(&values_actual, 5);

        let indices_expected = Data::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);
        assert_eq!(indices_expected, indices_actual);
    }
}
