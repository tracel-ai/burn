#[burn_tensor_testgen::testgen(sort_argsort)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn test_sort_1d_int() {
        let tensor = TestTensorInt::from([1, 4, 7, 2, 5, 6, 3, 0, 9, 8, 2, 8, -10, 42, 1000]);

        // Sort along dim=0
        let values = tensor.sort(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([-10, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 42, 1000]);
        assert_eq!(values_expected, values_actual);
    }

    #[test]
    fn test_argsort_1d_int() {
        let tensor = TestTensorInt::from([1, 4, 7, 2, 5, 6, 3, 0, 9, 8, -10, 42, 1000]);

        // Sort along dim=0
        let indices = tensor.argsort(0);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([10, 7, 0, 3, 6, 1, 4, 5, 2, 9, 8, 11, 12]);
        assert_eq!(indices_expected, indices_actual);
    }

    #[test]
    fn test_sort_with_indices_descending_int() {
        // 1D
        let tensor = TestTensorInt::from([1, 4, 7, 2, 5, 6, 3, 0, 9, 8, -10, 42, 1000]);

        // Sort along dim=0
        let (values, indices) = tensor.sort_descending_with_indices(0);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([1000, 42, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -10]);
        assert_eq!(values_expected, values_actual);

        let indices_expected = Data::from([12, 11, 8, 9, 2, 5, 4, 1, 6, 3, 0, 7, 10]);
        assert_eq!(indices_expected, indices_actual);

        // 2D
        let tensor = TestTensorInt::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        // Sort along dim=1
        let (values, indices) = tensor.sort_descending_with_indices(1);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([[[2, 5, 7], [1, 4, 6]], [[8, 2, 9], [3, 0, 8]]]);
        assert_eq!(values_expected, values_actual);

        let indices_expected = Data::from([[[1, 1, 0], [0, 0, 1]], [[1, 1, 0], [0, 0, 1]]]);
        assert_eq!(indices_expected, indices_actual);
    }

    #[test]
    fn test_sort_int() {
        let tensor = TestTensorInt::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        // Sort along dim=0
        let values = tensor.clone().sort(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([[[1, 0, 7], [2, 2, 6]], [[3, 4, 9], [8, 5, 8]]]);
        assert_eq!(values_expected, values_actual);

        // Sort along dim=1
        let values = tensor.clone().sort(1);
        let values_actual = values.into_data();

        let values_expected = Data::from([[[1, 4, 6], [2, 5, 7]], [[3, 0, 8], [8, 2, 9]]]);
        assert_eq!(values_expected, values_actual);

        // Sort along dim=2
        let values = tensor.sort(2);
        let values_actual = values.into_data();

        let values_expected = Data::from([[[1, 4, 7], [2, 5, 6]], [[0, 3, 9], [2, 8, 8]]]);
        assert_eq!(values_expected, values_actual);
    }

    #[test]
    fn test_sort_with_indices_int() {
        let tensor = TestTensorInt::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        // Sort along dim=0
        let (values, indices) = tensor.clone().sort_with_indices(0);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([[[1, 0, 7], [2, 2, 6]], [[3, 4, 9], [8, 5, 8]]]);
        assert_eq!(values_expected, values_actual);

        let indices_expected = Data::from([[[0, 1, 0], [0, 1, 0]], [[1, 0, 1], [1, 0, 1]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=1
        let (values, indices) = tensor.clone().sort_with_indices(1);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([[[1, 4, 6], [2, 5, 7]], [[3, 0, 8], [8, 2, 9]]]);
        assert_eq!(values_expected, values_actual);

        let indices_expected = Data::from([[[0, 0, 1], [1, 1, 0]], [[0, 0, 1], [1, 1, 0]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=2
        let (values, indices) = tensor.sort_with_indices(2);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([[[1, 4, 7], [2, 5, 6]], [[0, 3, 9], [2, 8, 8]]]);
        assert_eq!(values_expected, values_actual);

        // unstable sort could return [1, 0, 2] or [1, 2, 0] since it doesn't guarantee original order
        let indices_expected = Data::from([[[0, 1, 2], [0, 1, 2]], [[1, 0, 2], [1, 0, 2]]]);
        if indices_expected != indices_actual {
            assert_eq!(
                Data::from([[[0, 1, 2], [0, 1, 2]], [[1, 0, 2], [1, 2, 0]]]),
                indices_actual
            );
        }
    }

    #[test]
    fn test_argsort_int() {
        let tensor = TestTensorInt::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 8]]]);

        // Sort along dim=0
        let indices = tensor.clone().argsort(0);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([[[0, 1, 0], [0, 1, 0]], [[1, 0, 1], [1, 0, 1]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=1
        let indices = tensor.clone().argsort(1);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([[[0, 0, 1], [1, 1, 0]], [[0, 0, 1], [1, 1, 0]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=2
        let indices = tensor.argsort(2);
        let indices_actual = indices.into_data();

        // unstable sort could return [1, 0, 2] or [1, 2, 0] since it doesn't guarantee original order
        let indices_expected = Data::from([[[0, 1, 2], [0, 1, 2]], [[1, 0, 2], [1, 0, 2]]]);
        if indices_expected != indices_actual {
            assert_eq!(
                Data::from([[[0, 1, 2], [0, 1, 2]], [[1, 0, 2], [1, 2, 0]]]),
                indices_actual
            );
        }
    }

    #[test]
    fn test_sort_1d_float() {
        let tensor = TestTensor::from([
            0.5, 1.2, -0.21, 0., 2.1, 0.94, -0.3, 2.3, 199.412, 4., 0.99, 3., -8.1,
        ]);

        // Sort along dim=0
        let values = tensor.sort(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([
            -8.1, -0.3, -0.21, 0., 0.5, 0.94, 0.99, 1.2, 2.1, 2.3, 3., 4., 199.412,
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);
    }

    #[test]
    fn test_argsort_1d_float() {
        let tensor = TestTensor::from([
            0.5, 1.2, -0.21, 0., 2.1, 0.94, -0.3, 2.3, 199.412, 4., 0.99, 3., -8.1,
        ]);

        // Sort along dim=0
        let indices = tensor.argsort(0);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([12, 6, 2, 3, 0, 5, 10, 1, 4, 7, 11, 9, 8]);
        assert_eq!(indices_expected, indices_actual);
    }

    #[test]
    fn test_sort_with_indices_descending_float() {
        // 1D
        let tensor = TestTensor::from([
            0.5, 1.2, -0.21, 0., 2.1, 0.94, -0.3, 2.3, 199.412, 4., 0.99, 3., -8.1,
        ]);

        // Sort along dim=0
        let (values, indices) = tensor.sort_descending_with_indices(0);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([
            199.412, 4., 3., 2.3, 2.1, 1.2, 0.99, 0.94, 0.5, 0., -0.21, -0.3, -8.1,
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        let indices_expected = Data::from([8, 9, 11, 7, 4, 1, 10, 5, 0, 3, 2, 6, 12]);
        assert_eq!(indices_expected, indices_actual);

        // 2D
        let tensor = TestTensor::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
            [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
        ]);

        // Sort along dim=1
        let (values, indices) = tensor.sort_descending_with_indices(1);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([
            [[0., 2.1, 0.94], [-0.5, 1.2, -0.21]],
            [[0.99, 3., 4.], [-0.3, 2.3, -8.1]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        let indices_expected = Data::from([[[1, 1, 1], [0, 0, 0]], [[1, 1, 0], [0, 0, 1]]]);
        assert_eq!(indices_expected, indices_actual);
    }

    #[test]
    fn test_sort_float() {
        let tensor = TestTensor::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
            [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
        ]);

        // Sort along dim=0
        let values = tensor.clone().sort(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, -8.1]],
            [[-0.3, 2.3, 4.], [0.99, 3., 0.94]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        // Sort along dim=1
        let values = tensor.clone().sort(1);
        let values_actual = values.into_data();

        let values_expected = Data::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
            [[-0.3, 2.3, -8.1], [0.99, 3., 4.]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        // Sort along dim=2
        let values = tensor.sort(2);
        let values_actual = values.into_data();

        let values_expected = Data::from([
            [[-0.5, -0.21, 1.2], [0., 0.94, 2.1]],
            [[-0.3, 2.3, 4.], [-8.1, 0.99, 3.]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);
    }

    #[test]
    fn test_sort_with_indices_float() {
        let tensor = TestTensor::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
            [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
        ]);

        // Sort along dim=0
        let (values, indices) = tensor.clone().sort_with_indices(0);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, -8.1]],
            [[-0.3, 2.3, 4.], [0.99, 3., 0.94]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        let indices_expected = Data::from([[[0, 0, 0], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=1
        let (values, indices) = tensor.clone().sort_with_indices(1);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
            [[-0.3, 2.3, -8.1], [0.99, 3., 4.]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        let indices_expected = Data::from([[[0, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 0]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=2
        let (values, indices) = tensor.sort_with_indices(2);
        let values_actual = values.into_data();
        let indices_actual = indices.into_data();

        let values_expected = Data::from([
            [[-0.5, -0.21, 1.2], [0., 0.94, 2.1]],
            [[-0.3, 2.3, 4.], [-8.1, 0.99, 3.]],
        ]);
        values_expected.assert_approx_eq(&values_actual, 5);

        let indices_expected = Data::from([[[0, 2, 1], [0, 2, 1]], [[0, 1, 2], [2, 0, 1]]]);
        assert_eq!(indices_expected, indices_actual);
    }

    #[test]
    fn test_argsort_float() {
        let tensor = TestTensor::from([
            [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
            [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
        ]);

        // Sort along dim=0
        let indices = tensor.clone().argsort(0);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([[[0, 0, 0], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=1
        let indices = tensor.clone().argsort(1);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([[[0, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 0]]]);
        assert_eq!(indices_expected, indices_actual);

        // Sort along dim=2
        let indices = tensor.argsort(2);
        let indices_actual = indices.into_data();

        let indices_expected = Data::from([[[0, 2, 1], [0, 2, 1]], [[0, 1, 2], [2, 0, 1]]]);
        assert_eq!(indices_expected, indices_actual);
    }

    #[test]
    fn test_sort_float_nan() {
        let tensor = TestTensor::from([[-0.5, f32::NAN], [0., 0.94], [-0.3, f32::NAN]]);

        // Sort along dim=0
        let values = tensor.sort(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([[-0.5, 0.94], [-0.3, f32::NAN], [0., f32::NAN]]);
        values_expected.assert_approx_eq(&values_actual, 5);
    }

    #[test]
    fn test_sort_descending_1d() {
        let tensor = TestTensorInt::from([1, 2, 3, 4, 5]);

        // Sort along dim=0
        let values = tensor.sort_descending(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([5, 4, 3, 2, 1]);
        assert_eq!(values_expected, values_actual);

        let tensor = TestTensor::from([1., 2., 3., 4., 5.]);

        // Sort along dim=0
        let values = tensor.sort_descending(0);
        let values_actual = values.into_data();

        let values_expected = Data::from([5., 4., 3., 2., 1.]);
        values_expected.assert_approx_eq(&values_actual, 5);
    }
}
