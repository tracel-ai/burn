use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_sort_1d_float() {
    let tensor = TestTensor::<1>::from([
        0.5, 1.2, -0.21, 0., 2.1, 0.94, -0.3, 2.3, 199.412, 4., 0.99, 3., -8.1,
    ]);

    // Sort along dim=0
    let values = tensor.sort(0);

    let values_expected = TensorData::from([
        -8.1, -0.3, -0.21, 0., 0.5, 0.94, 0.99, 1.2, 2.1, 2.3, 3., 4., 199.412,
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());
}

#[test]
fn test_argsort_1d_float() {
    let tensor = TestTensor::<1>::from([
        0.5, 1.2, -0.21, 0., 2.1, 0.94, -0.3, 2.3, 199.412, 4., 0.99, 3., -8.1,
    ]);

    // Sort along dim=0
    let indices = tensor.argsort(0);

    let indices_expected = TensorData::from([12, 6, 2, 3, 0, 5, 10, 1, 4, 7, 11, 9, 8]);
    indices.into_data().assert_eq(&indices_expected, false);
}

#[test]
fn test_sort_with_indices_descending_float() {
    // 1D
    let tensor = TestTensor::<1>::from([
        0.5, 1.2, -0.21, 0., 2.1, 0.94, -0.3, 2.3, 199.412, 4., 0.99, 3., -8.1,
    ]);

    // Sort along dim=0
    let (values, indices) = tensor.sort_descending_with_indices(0);

    let values_expected = TensorData::from([
        199.412, 4., 3., 2.3, 2.1, 1.2, 0.99, 0.94, 0.5, 0., -0.21, -0.3, -8.1,
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    let indices_expected = TensorData::from([8, 9, 11, 7, 4, 1, 10, 5, 0, 3, 2, 6, 12]);
    indices.into_data().assert_eq(&indices_expected, false);

    // 2D
    let tensor = TestTensor::<3>::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
        [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
    ]);

    // Sort along dim=1
    let (values, indices) = tensor.sort_descending_with_indices(1);

    let values_expected = TensorData::from([
        [[0., 2.1, 0.94], [-0.5, 1.2, -0.21]],
        [[0.99, 3., 4.], [-0.3, 2.3, -8.1]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    let indices_expected = TensorData::from([[[1, 1, 1], [0, 0, 0]], [[1, 1, 0], [0, 0, 1]]]);
    indices.into_data().assert_eq(&indices_expected, false);
}

#[test]
fn test_sort_float() {
    let tensor = TestTensor::<3>::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
        [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
    ]);

    // Sort along dim=0
    let values = tensor.clone().sort(0);

    let values_expected = TensorData::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, -8.1]],
        [[-0.3, 2.3, 4.], [0.99, 3., 0.94]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    // Sort along dim=1
    let values = tensor.clone().sort(1);

    let values_expected = TensorData::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
        [[-0.3, 2.3, -8.1], [0.99, 3., 4.]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    // Sort along dim=2
    let values = tensor.sort(2);

    let values_expected = TensorData::from([
        [[-0.5, -0.21, 1.2], [0., 0.94, 2.1]],
        [[-0.3, 2.3, 4.], [-8.1, 0.99, 3.]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());
}

#[test]
fn test_sort_with_indices_float() {
    let tensor = TestTensor::<3>::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
        [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
    ]);

    // Sort along dim=0
    let (values, indices) = tensor.clone().sort_with_indices(0);
    let values_expected = TensorData::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, -8.1]],
        [[-0.3, 2.3, 4.], [0.99, 3., 0.94]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    let indices_expected = TensorData::from([[[0, 0, 0], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]);
    indices.into_data().assert_eq(&indices_expected, false);

    // Sort along dim=1
    let (values, indices) = tensor.clone().sort_with_indices(1);

    let values_expected = TensorData::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
        [[-0.3, 2.3, -8.1], [0.99, 3., 4.]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    let indices_expected = TensorData::from([[[0, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 0]]]);
    indices.into_data().assert_eq(&indices_expected, false);

    // Sort along dim=2
    let (values, indices) = tensor.sort_with_indices(2);

    let values_expected = TensorData::from([
        [[-0.5, -0.21, 1.2], [0., 0.94, 2.1]],
        [[-0.3, 2.3, 4.], [-8.1, 0.99, 3.]],
    ]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    let indices_expected = TensorData::from([[[0, 2, 1], [0, 2, 1]], [[0, 1, 2], [2, 0, 1]]]);
    indices.into_data().assert_eq(&indices_expected, false);
}

#[test]
fn test_argsort_float() {
    let tensor = TestTensor::<3>::from([
        [[-0.5, 1.2, -0.21], [0., 2.1, 0.94]],
        [[-0.3, 2.3, 4.], [0.99, 3., -8.1]],
    ]);

    // Sort along dim=0
    let indices = tensor.clone().argsort(0);

    let indices_expected = TensorData::from([[[0, 0, 0], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]);
    indices.into_data().assert_eq(&indices_expected, false);

    // Sort along dim=1
    let indices = tensor.clone().argsort(1);

    let indices_expected = TensorData::from([[[0, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 0]]]);
    indices.into_data().assert_eq(&indices_expected, false);

    // Sort along dim=2
    let indices = tensor.argsort(2);

    let indices_expected = TensorData::from([[[0, 2, 1], [0, 2, 1]], [[0, 1, 2], [2, 0, 1]]]);
    indices.into_data().assert_eq(&indices_expected, false);
}

#[test]
fn test_sort_float_nan() {
    let tensor = TestTensor::<2>::from([[-0.5, f32::NAN], [0., 0.94], [-0.3, f32::NAN]]);

    // Sort along dim=0
    let values = tensor.sort(0);

    let values_expected = TensorData::from([[-0.5, 0.94], [-0.3, f32::NAN], [0., f32::NAN]]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());
}

#[test]
fn test_sort_descending_1d() {
    let tensor = TestTensor::<1>::from([1., 2., 3., 4., 5.]);

    // Sort along dim=0
    let values = tensor.sort_descending(0);

    let values_expected = TensorData::from([5., 4., 3., 2., 1.]);
    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());
}
