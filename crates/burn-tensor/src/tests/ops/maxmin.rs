#[burn_tensor_testgen::testgen(maxmin)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Tensor, TensorData};

    #[test]
    fn test_max_dim_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.max_dim(1);
        let expected =
            TensorData::from([[2.], [5.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_max_dim_with_indices_2d_with_dim_0th() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let (output, index) = tensor.max_dim_with_indices(0);

        let output_expected =
            TensorData::from([[3., 4., 5.]]).convert::<<TestBackend as Backend>::FloatElem>();
        let index_expected =
            TensorData::from([[1, 1, 1]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&output_expected, true);
        index.into_data().assert_eq(&index_expected, true);
    }

    #[test]
    fn test_max_dim_with_indices_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let (output, index) = tensor.max_dim_with_indices(1);

        let output_expected =
            TensorData::from([[2.], [5.]]).convert::<<TestBackend as Backend>::FloatElem>();
        let index_expected =
            TensorData::from([[2], [2]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&output_expected, true);
        index.into_data().assert_eq(&index_expected, true);
    }

    #[test]
    fn test_min_dim_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.min_dim(1);

        let expected =
            TensorData::from([[0.], [3.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_min_dim_with_indices_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let (output, index) = tensor.min_dim_with_indices(1);

        let output_expected =
            TensorData::from([[0.], [3.]]).convert::<<TestBackend as Backend>::FloatElem>();
        let index_expected =
            TensorData::from([[0], [0]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&output_expected, true);
        index.into_data().assert_eq(&index_expected, true);
    }

    #[test]
    fn test_sum_dim_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.clone().sum_dim(1);
        let expected =
            TensorData::from([[3.], [12.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);

        let output = tensor.sum_dim(0);
        let expected =
            TensorData::from([[3., 5., 7.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_mean_dim_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.clone().mean_dim(1);
        let expected =
            TensorData::from([[1.], [4.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_approx_eq(&expected, 3);

        let output = tensor.mean_dim(0);
        let expected =
            TensorData::from([[1.5, 2.5, 3.5]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_min_dim_2d_with_0th_dim() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.min_dim(0);
        let expected =
            TensorData::from([[0., 1., 2.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_max_dim_2d_with_0th_dim() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.max_dim(0);
        let expected =
            TensorData::from([[3., 4., 5.]]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_min_dim_with_indices_2d_with_0th_dim() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let (output, index) = tensor.min_dim_with_indices(0);

        let output_expected =
            TensorData::from([[0., 1., 2.]]).convert::<<TestBackend as Backend>::FloatElem>();
        let index_expected =
            TensorData::from([[0, 0, 0]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&output_expected, true);
        index.into_data().assert_eq(&index_expected, true);
    }

    #[test]
    fn test_maximum_pair() {
        let a = TestTensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &Default::default());
        let b = TestTensor::from_floats([2.0, 1.0, 4.0, 5.0], &Default::default());

        let output = a.max_pair(b);
        let expected =
            TensorData::from([2.0, 2.0, 4.0, 5.0]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_minimum_pair() {
        let a = TestTensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &Default::default());
        let b = TestTensor::from_floats([2.0, 1.0, 4.0, 5.0], &Default::default());

        let output = a.min_pair(b);
        let expected =
            TensorData::from([1.0, 1.0, 3.0, 4.0]).convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_approx_eq(&expected, 1);
    }
}
