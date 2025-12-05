use super::qtensor::*;
use super::*;
use burn_tensor::IndexingUpdateOp;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_select_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0]);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &Default::default());

    let output = tensor.select(0, indices);
    let expected = TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_select_2d_dim0_same_num_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indices = TestTensorInt::from_data([1, 0], &Default::default());

    let output = tensor.select(0, indices);
    let expected = TensorData::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_select_2d_dim0_more_num_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indices = TestTensorInt::from_data([1, 0, 1, 1], &Default::default());

    let output = tensor.select(0, indices);
    let expected = TensorData::from([
        [3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
    ]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_select_2d_dim1() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &Default::default());

    let output = tensor.select(1, indices);
    let expected = TensorData::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_select_assign_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0]);
    let values = QTensor::<TestBackend, 1>::int8([5.0, 4.0, 3.0, 2.0, 1.0]);
    let indices = TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &Default::default());

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([3.0, 12.0, 3.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_select_assign_2d_dim0() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let values = tensor.clone();
    let indices = TestTensorInt::from_data(TensorData::from([1, 0]), &Default::default());

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([[3.0, 5.0, 7.0], [3.0, 5.0, 7.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_select_assign_2d_dim1() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let values = tensor.clone();
    let indices = TestTensorInt::from_data(TensorData::from([1, 0, 2]), &Default::default());

    let output = tensor.select_assign(1, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([[1.0, 1.0, 4.0], [7.0, 7.0, 10.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
#[should_panic]
fn should_select_panic_invalid_dimension() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &Default::default());

    tensor.select(10, indices);
}
