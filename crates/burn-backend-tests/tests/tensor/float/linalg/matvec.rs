use super::*;
use burn_tensor::{TensorData, Tolerance, linalg};

#[test]
fn test_matvec_basic_float() {
    let device = Default::default();
    let matrix = TestTensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let vector = TestTensor::<1>::from_floats([5.0, 6.0], &device);

    let result = linalg::matvec::<TestBackend, 2, 1, _>(matrix, vector);
    let expected = TensorData::from([17.0, 39.0]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_matvec_basic_int() {
    let device = Default::default();
    let matrix = TestTensorInt::<2>::from_ints([[2, 0, -1], [1, 3, 2]], &device);
    let vector = TestTensorInt::<1>::from_ints([3, -2, 4], &device);

    let result = linalg::matvec::<TestBackend, 2, 1, _>(matrix, vector);
    let expected = TensorData::from([2, 5]);

    result.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matvec_batched() {
    let device = Default::default();
    let matrix = TestTensor::<3>::from_floats(
        [
            [[1.0, 0.0, 2.0], [3.0, 1.0, -1.0]],
            [[-2.0, 1.0, 0.0], [0.5, -1.5, 2.0]],
        ],
        &device,
    );
    let vector = TestTensor::<2>::from_floats([[1.0, -1.0, 0.5], [2.0, 0.0, -1.0]], &device);

    let result = linalg::matvec::<TestBackend, 3, 2, _>(matrix, vector);
    let expected = TensorData::from([[2.0, 1.5], [-4.0, -1.0]]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_matvec_vector_broadcasts_over_batches() {
    let device = Default::default();
    let matrix = TestTensor::<3>::from_floats(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, 0.0, 2.0], [3.0, 1.0, -2.0]],
        ],
        &device,
    );
    let vector = TestTensor::<2>::from_floats([[1.0, 0.0, -1.0]], &device);

    let result = linalg::matvec::<TestBackend, 3, 2, _>(matrix, vector);
    let expected = TensorData::from([[-2.0, -2.0], [-3.0, 5.0]]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_matvec_matrix_broadcasts_over_vector_batches() {
    let device = Default::default();
    let matrix = TestTensor::<3>::from_floats([[[1.0, 0.0, 2.0], [3.0, -1.0, 1.0]]], &device);
    let vector = TestTensor::<2>::from_floats([[2.0, 1.0, 0.0], [1.0, -1.0, 3.0]], &device);

    let result = linalg::matvec::<TestBackend, 3, 2, _>(matrix, vector);
    let expected = TensorData::from([[2.0, 5.0], [7.0, 7.0]]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[should_panic]
fn test_matvec_invalid_inner_dim_panics() {
    let device = Default::default();
    let matrix = TestTensor::<2>::zeros([2, 3], &device);
    let vector = TestTensor::<1>::zeros([4], &device);

    let _ = linalg::matvec::<TestBackend, 2, 1, _>(matrix, vector);
}

#[test]
#[should_panic]
fn test_matvec_mismatched_batches_panics() {
    let device = Default::default();
    let matrix = TestTensor::<3>::zeros([2, 3, 4], &device);
    let vector = TestTensor::<2>::zeros([3, 4], &device);

    let _ = linalg::matvec::<TestBackend, 3, 2, _>(matrix, vector);
}
