#[burn_tensor_testgen::testgen(maxmin)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_max_dim_2d() {
        let f =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        f.clone()
            .max_dim(0)
            .into_data()
            .assert_eq(&TensorData::from([[3., 4., 5.]]), false);

        f.clone()
            .max_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[2.], [5.]]), false);

        // Negative Index
        f.clone()
            .max_dim(-1)
            .into_data()
            .assert_eq(&TensorData::from([[2.], [5.]]), false);

        // Regression Test: https://github.com/tracel-ai/burn/issues/3139
        let z = f.clone().int();
        z.clone()
            .max_dim(0)
            .into_data()
            .assert_eq(&TensorData::from([[3, 4, 5]]).into(), false);
        z.clone()
            .max_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[2], [5]]).into(), false);
    }

    #[test]
    fn test_max_dims_2d() {
        let f =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        f.clone()
            .max_dims(&[0])
            .into_data()
            .assert_eq(&TensorData::from([[3., 4., 5.]]), false);

        f.clone()
            .max_dims(&[-2])
            .into_data()
            .assert_eq(&TensorData::from([[3., 4., 5.]]), false);

        f.clone()
            .max_dims(&[0, 1])
            .into_data()
            .assert_eq(&TensorData::from([[5.]]), false);
    }

    #[test]
    fn test_max_dim_with_indices_2d_with_dim_0th() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        // Positive, Negative Index
        for idx in [0, -2] {
            let (output, index) = tensor.clone().max_dim_with_indices(idx);

            let output_expected = TensorData::from([[3., 4., 5.]]);
            let index_expected = TensorData::from([[1, 1, 1]]);

            output.into_data().assert_eq(&output_expected, false);
            index.into_data().assert_eq(&index_expected, false);
        }
    }

    #[test]
    fn test_max_dim_with_indices_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let (output, index) = tensor.max_dim_with_indices(1);

        let output_expected = TensorData::from([[2.], [5.]]);
        let index_expected = TensorData::from([[2], [2]]);

        output.into_data().assert_eq(&output_expected, false);
        index.into_data().assert_eq(&index_expected, false);
    }

    #[test]
    fn test_max_dim_2d_with_0th_dim() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.max_dim(0);
        let expected = TensorData::from([[3., 4., 5.]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_max_pair() {
        let a = TestTensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &Default::default());
        let b = TestTensor::from_floats([2.0, 1.0, 4.0, 5.0], &Default::default());

        let output = a.max_pair(b);
        let expected = TensorData::from([2.0, 2.0, 4.0, 5.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_min_dim_2d() {
        let f =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        f.clone()
            .min_dim(0)
            .into_data()
            .assert_eq(&TensorData::from([[0., 1., 2.]]), false);

        f.clone()
            .min_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[0.], [3.]]), false);

        // Negative Index
        f.clone()
            .min_dim(-1)
            .into_data()
            .assert_eq(&TensorData::from([[0.], [3.]]), false);

        // Regression Test: https://github.com/tracel-ai/burn/issues/3139
        let z = f.int();
        z.clone()
            .min_dim(0)
            .into_data()
            .assert_eq(&TensorData::from([[0, 1, 2]]).into(), false);
        z.clone()
            .min_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[0], [3]]).into(), false);
    }

    #[test]
    fn test_min_dims_2d() {
        let f =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        f.clone()
            .min_dims(&[0])
            .into_data()
            .assert_eq(&TensorData::from([[0., 1., 2.]]), false);

        f.clone()
            .min_dims(&[-2])
            .into_data()
            .assert_eq(&TensorData::from([[0., 1., 2.]]), false);

        f.clone()
            .min_dims(&[0, 1])
            .into_data()
            .assert_eq(&TensorData::from([[0.]]), false);
    }

    #[test]
    fn test_min_dim_with_indices_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let (output, index) = tensor.min_dim_with_indices(1);

        let output_expected = TensorData::from([[0.], [3.]]);
        let index_expected = TensorData::from([[0], [0]]);

        output.into_data().assert_eq(&output_expected, false);
        index.into_data().assert_eq(&index_expected, false);
    }

    #[test]
    fn test_min_dim_2d_with_0th_dim() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.min_dim(0);
        let expected = TensorData::from([[0., 1., 2.]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_min_dim_with_indices_2d_with_0th_dim() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        // Positive, Negative Index
        for idx in [0, -2] {
            let (output, index) = tensor.clone().min_dim_with_indices(idx);

            let output_expected = TensorData::from([[0., 1., 2.]]);
            let index_expected = TensorData::from([[0, 0, 0]]);

            output.into_data().assert_eq(&output_expected, false);
            index.into_data().assert_eq(&index_expected, false);
        }
    }

    #[test]
    fn test_min_pair() {
        let a = TestTensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &Default::default());
        let b = TestTensor::from_floats([2.0, 1.0, 4.0, 5.0], &Default::default());

        let output = a.min_pair(b);
        let expected = TensorData::from([1.0, 1.0, 3.0, 4.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_max_abs() {
        let tensor =
            TestTensor::<2>::from_floats([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

        let output = tensor.max_abs();
        let expected = TensorData::from([6.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_max_abs_dim_2d_dim_0() {
        let tensor =
            TestTensor::<2>::from_floats([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

        let output = tensor.clone().max_abs_dim(0);
        let expected = TensorData::from([[5., 6., 2.]]);
        output.into_data().assert_eq(&expected, false);

        // Negative Index
        let output = tensor.clone().max_abs_dim(-2);
        let expected = TensorData::from([[5., 6., 2.]]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_max_abs_dim_2d_dim_1() {
        let tensor =
            TestTensor::<2>::from_floats([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

        let output = tensor.max_abs_dim(1);
        let expected = TensorData::from([[2.], [6.]]);

        output.into_data().assert_eq(&expected, false);
    }
}
