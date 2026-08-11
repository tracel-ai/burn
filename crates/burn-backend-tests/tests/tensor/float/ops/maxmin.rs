use super::*;
use burn_tensor::{ElementConversion, TensorData};

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

#[test]
fn test_whole_max_min_finite_and_infinite_values() {
    let tensor = TestTensor::<1>::from([f32::NEG_INFINITY, -2.0, 7.0, 7.0, f32::INFINITY]);

    tensor
        .clone()
        .max()
        .into_data()
        .assert_eq(&TensorData::from([f32::INFINITY]), false);
    tensor
        .min()
        .into_data()
        .assert_eq(&TensorData::from([f32::NEG_INFINITY]), false);
}

#[test]
fn test_whole_max_min_integer_control() {
    let tensor = TestTensor::<1>::from([3.0, -2.0, 7.0, 1.0]).int();

    tensor
        .clone()
        .max()
        .into_data()
        .assert_eq(&TensorData::from([7]), false);
    tensor
        .min()
        .into_data()
        .assert_eq(&TensorData::from([-2]), false);
}

#[test]
fn test_whole_max_min_nan_propagation_by_position() {
    for values in [
        [f32::NAN, 1.0, 2.0],
        [1.0, f32::NAN, 2.0],
        [1.0, 2.0, f32::NAN],
    ] {
        let tensor = TestTensor::<1>::from(values);

        let max = tensor.clone().max().into_data();
        assert!(max.as_slice::<FloatElem>().unwrap()[0].is_nan());

        let min = tensor.min().into_data();
        assert!(min.as_slice::<FloatElem>().unwrap()[0].is_nan());
    }
}

#[test]
fn test_whole_max_min_nan_non_contiguous_view() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 2.0], [3.0, 4.0, 5.0]])
        .narrow(0, 0, 1)
        .swap_dims(0, 1);

    let max = tensor.clone().max().into_data();
    assert!(max.as_slice::<FloatElem>().unwrap()[0].is_nan());

    let min = tensor.min().into_data();
    assert!(min.as_slice::<FloatElem>().unwrap()[0].is_nan());
}

#[test]
fn test_whole_max_min_ignore_nan_outside_logical_view() {
    let tensor = TestTensor::<2>::from([[f32::NAN, 1.0, 2.0], [3.0, 4.0, 5.0]])
        .narrow(0, 1, 1)
        .swap_dims(0, 1);

    tensor
        .clone()
        .max()
        .into_data()
        .assert_eq(&TensorData::from([5.0]), false);
    tensor
        .min()
        .into_data()
        .assert_eq(&TensorData::from([3.0]), false);
}

#[test]
fn test_whole_max_min_expanded_view_uses_only_logical_elements() {
    let tensor = TestTensor::<2>::from([[f32::NAN, 1.0], [2.0, 3.0]])
        .narrow(0, 1, 1)
        .expand([2, 2]);

    tensor
        .clone()
        .max()
        .into_data()
        .assert_eq(&TensorData::from([3.0]), false);
    tensor
        .min()
        .into_data()
        .assert_eq(&TensorData::from([2.0]), false);
}

#[test]
fn test_whole_max_abs_nan_propagation() {
    let tensor = TestTensor::<1>::from([-3.0, f32::NAN, 2.0]);
    let output = tensor.max_abs().into_data();
    assert!(output.as_slice::<FloatElem>().unwrap()[0].is_nan());
}

#[test]
fn test_max_abs_dim_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, -3.0], [1.0, -4.0, 3.0]]);
    let output = tensor.max_abs_dim(1).into_data();
    let values = output.as_slice::<FloatElem>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1], 4.0f32.elem::<FloatElem>());
}

#[cfg(any(feature = "flex", feature = "ndarray"))]
#[test]
fn test_whole_max_min_nan_f64_cpu_backends() {
    let tensor = TestTensor::<1>::from([1.0, f32::NAN, 2.0]).cast(burn_tensor::DType::F64);

    assert!(tensor.clone().max().into_data().as_slice::<f64>().unwrap()[0].is_nan());
    assert!(tensor.min().into_data().as_slice::<f64>().unwrap()[0].is_nan());
}

#[cfg(feature = "flex")]
#[test]
fn test_whole_max_min_nan_flex_half_dtypes() {
    for dtype in [burn_tensor::DType::F16, burn_tensor::DType::BF16] {
        let tensor = TestTensor::<1>::from([1.0, f32::NAN, 2.0]).cast(dtype);

        for output in [tensor.clone().max(), tensor.min()] {
            let output = output.into_data().convert::<f32>();
            assert!(output.as_slice::<f32>().unwrap()[0].is_nan());
        }
    }
}

// All burn backends should propagate NaN from min/max. See issue #4814.
#[test]
fn test_max_min_dim_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 3.0]]);

    for output in [tensor.clone().max_dim(1), tensor.min_dim(1)] {
        let data = output.into_data();
        assert!(data.as_slice::<FloatElem>().unwrap()[0].is_nan());
    }
}

#[test]
fn test_max_min_dim_with_indices_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, -3.0]]);

    for (values, indices) in [
        tensor.clone().max_dim_with_indices(1),
        tensor.min_dim_with_indices(1),
    ] {
        let values = values.into_data();
        assert!(values.as_slice::<FloatElem>().unwrap()[0].is_nan());
        indices
            .into_data()
            .assert_eq(&TensorData::from([[1]]), false);
    }
}
