use super::*;
use burn_tensor::TensorData;

#[test]
fn test_max_dim_2d() {
    let f = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

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
        .assert_eq(&TensorData::from([[3, 4, 5]]), false);
    z.clone()
        .max_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[2], [5]]), false);
}

#[test]
fn test_max_dims_2d() {
    let f = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

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
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

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
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    let (output, index) = tensor.max_dim_with_indices(1);

    let output_expected = TensorData::from([[2.], [5.]]);
    let index_expected = TensorData::from([[2], [2]]);

    output.into_data().assert_eq(&output_expected, false);
    index.into_data().assert_eq(&index_expected, false);
}

#[test]
fn test_max_dim_2d_with_0th_dim() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    let output = tensor.max_dim(0);
    let expected = TensorData::from([[3., 4., 5.]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_max_pair() {
    let a = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &Default::default());
    let b = TestTensor::from_data([2.0, 1.0, 4.0, 5.0], &Default::default());

    let output = a.max_pair(b);
    let expected = TensorData::from([2.0, 2.0, 4.0, 5.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_min_dim_2d() {
    let f = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

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
        .assert_eq(&TensorData::from([[0, 1, 2]]), false);
    z.clone()
        .min_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[0], [3]]), false);
}

#[test]
fn test_min_dims_2d() {
    let f = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

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
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    let (output, index) = tensor.min_dim_with_indices(1);

    let output_expected = TensorData::from([[0.], [3.]]);
    let index_expected = TensorData::from([[0], [0]]);

    output.into_data().assert_eq(&output_expected, false);
    index.into_data().assert_eq(&index_expected, false);
}

#[test]
fn test_min_dim_2d_with_0th_dim() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    let output = tensor.min_dim(0);
    let expected = TensorData::from([[0., 1., 2.]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_min_dim_with_indices_2d_with_0th_dim() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

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
    let a = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &Default::default());
    let b = TestTensor::from_data([2.0, 1.0, 4.0, 5.0], &Default::default());

    let output = a.min_pair(b);
    let expected = TensorData::from([1.0, 1.0, 3.0, 4.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_max_abs() {
    let tensor = TestTensor::<2>::from_data([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

    let output = tensor.max_abs();
    let expected = TensorData::from([6.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_max_abs_dim_2d_dim_0() {
    let tensor = TestTensor::<2>::from_data([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

    let output = tensor.clone().max_abs_dim(0);
    let expected = TensorData::from([[5., 6., 2.]]);
    output.into_data().assert_eq(&expected, false);

    // Negative Index
    let output = tensor.clone().max_abs_dim(-2);
    let expected = TensorData::from([[5., 6., 2.]]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_max_abs_dims_2d() {
    let tensor = TestTensor::<2>::from_data([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

    tensor
        .clone()
        .max_abs_dims(&[0])
        .into_data()
        .assert_eq(&TensorData::from([[5., 6., 2.]]), false);

    tensor
        .clone()
        .max_abs_dims(&[-2])
        .into_data()
        .assert_eq(&TensorData::from([[5., 6., 2.]]), false);

    tensor
        .clone()
        .max_abs_dims(&[0, 1])
        .into_data()
        .assert_eq(&TensorData::from([[6.]]), false);
}

#[test]
fn test_max_abs_dim_2d_dim_1() {
    let tensor = TestTensor::<2>::from_data([[0., 1., -2.], [-5., 6., 1.]], &Default::default());

    let output = tensor.max_abs_dim(1);
    let expected = TensorData::from([[2.], [6.]]);

    output.into_data().assert_eq(&expected, false);
}

// NaN-propagation tests below. Only run when the `flex` backend feature
// is active, because flex is the only burn backend that currently
// propagates NaN from min/max (matching PyTorch/NumPy/JAX/TF). ndarray
// and the cubecl backends follow IEEE 754 min/max and drop NaN. The
// positive-gate form (rather than excluding specific backends) is used
// because the default-feature CI build selects a backend transitively
// without setting any of its identifying feature flags on
// burn-backend-tests, so a negative gate would still run the test on a
// NaN-dropping backend. See issue #4814.
#[cfg(feature = "flex")]
#[test]
fn test_max_dim_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 3.0]]);
    let data = tensor.max_dim(1).into_data();
    let values = data.as_slice::<FloatElem>().unwrap();
    assert!(values[0].is_nan());
}

#[cfg(feature = "flex")]
#[test]
fn test_min_dim_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 3.0]]);
    let data = tensor.min_dim(1).into_data();
    let values = data.as_slice::<FloatElem>().unwrap();
    assert!(values[0].is_nan());
}

#[cfg(feature = "flex")]
#[test]
fn test_max_dim_with_indices_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 3.0]]);
    let (values, indices) = tensor.max_dim_with_indices(1);
    let vdata = values.into_data();
    let slice = vdata.as_slice::<FloatElem>().unwrap();
    assert!(slice[0].is_nan());
    indices
        .into_data()
        .assert_eq(&TensorData::from([[1]]), false);
}
