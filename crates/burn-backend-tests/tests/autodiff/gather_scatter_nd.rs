use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn test_scatter_nd_add_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[10.0, 20.0, 30.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    // scatter_nd_add: data[1, :] += values[0, :]
    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Add);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data = all ones (identity for add)
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        false,
    );
    // grad_values = gather_nd(ones, indices) = ones at row 1
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0]]), false);
}

#[test]
fn test_scatter_nd_assign_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[10.0, 20.0, 30.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    // scatter_nd (assign): data[1, :] = values[0, :]
    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Assign);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data: all ones except row 1 is zeroed out (overwritten positions get no gradient)
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
    // grad_values = gather_nd(ones, indices) = ones at row 1
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0]]), false);
}

#[test]
fn test_gather_nd_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0], [2]]), &device);

    // gather_nd: extract rows 0 and 2
    let result: TestTensor<2> = data.clone().gather_nd(indices);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();

    // grad_data: scatter_nd(zeros, indices, ones, Add) -> row 0 and 2 get 1s, row 1 stays 0
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_add_grad_k_equals_d() {
    // K=D: scalar-level indexing (each index tuple selects a single element)
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device).require_grad();
    let values: TestTensor<1> =
        TestTensor::from_data(TensorData::from([10.0, 20.0]), &device).require_grad();
    // indices shape: [2, 2] => K=2=D, M=2, DV = 1+2-2 = 1
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1], [1, 0]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Add);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data = all ones (identity for add)
    grad_data
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0], [1.0, 1.0]]), false);
    // grad_values = gather_nd(ones, indices) = [1.0, 1.0]
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([1.0, 1.0]), false);
}

#[test]
fn test_gather_nd_grad_k_equals_d() {
    // K=D: scalar-level gather
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device).require_grad();
    // indices shape: [2, 2] => K=2=D
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1], [1, 0]]), &device);

    let result: TestTensor<1> = data.clone().gather_nd(indices);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();

    // grad = scatter_nd(zeros, indices, ones, Add)
    // [0,1] => data[0][1] gets 1, [1,0] => data[1][0] gets 1
    grad_data
        .to_data()
        .assert_eq(&TensorData::from([[0.0, 1.0], [1.0, 0.0]]), false);
}

#[test]
fn test_scatter_nd_mul_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[10.0, 20.0, 30.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    // scatter_nd_mul: data[1, :] *= values[0, :]
    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Mul);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data = grad * scatter_nd(ones, indices, values, Assign)
    //   row 1 gets the values, others stay at 1.
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [10.0, 20.0, 30.0], [1.0, 1.0, 1.0]]),
        false,
    );
    // grad_values = gather_nd(grad, indices) * gather_nd(data, indices)
    //             = ones * data[1, :] = [4.0, 5.0, 6.0]
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[4.0, 5.0, 6.0]]), false);
}

#[test]
fn test_scatter_nd_max_grad_data_wins() {
    // values < data at scattered positions: gradient flows fully to data.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0, 3.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Max);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // data wins everywhere on row 1, so grad_data = ones (full flow).
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        false,
    );
    // values lose, so grad_values is zero.
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[0.0, 0.0, 0.0]]), false);
}

#[test]
fn test_scatter_nd_max_grad_values_win() {
    // values > data at scattered positions: grad_data zeroed at scattered positions.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[10.0, 20.0, 30.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Max);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data zeroed at row 1 (values won), ones elsewhere.
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0]]), false);
}

#[test]
fn test_scatter_nd_max_grad_tie() {
    // Mixed: column 0 -> data wins, column 1 -> tie (both get grad), column 2 -> values wins.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 5.0, 10.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Max);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // Row 1: data_won = [T, T, F], values_won = [F, T, T] (tie at col 1 contributes to both).
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[0.0, 1.0, 1.0]]), false);
}

#[test]
fn test_scatter_nd_min_grad() {
    // Min mirror of test_scatter_nd_max_grad_values_win: values < data => values win for Min.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0, 3.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Min);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0]]), false);
}

#[test]
fn test_scatter_nd_min_grad_tie() {
    // Mixed: column 0 -> values win, column 1 -> tie, column 2 -> data wins.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 5.0, 10.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Min);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // Row 1: data_won = [F, T, T], values_won = [T, T, F].
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        false,
    );
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 0.0]]), false);
}

#[test]
fn test_scatter_nd_max_grad_k_equals_d() {
    // K=D scalar-level Max: each index targets one element.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device).require_grad();
    let values: TestTensor<1> =
        TestTensor::from_data(TensorData::from([10.0, 1.0]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1], [1, 0]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Max);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    grad_data
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 0.0], [1.0, 1.0]]), false);
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([1.0, 0.0]), false);
}

#[test]
fn test_scatter_nd_mul_grad_k_equals_d() {
    // K=D scalar-level Mul: each index targets one element.
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device).require_grad();
    let values: TestTensor<1> =
        TestTensor::from_data(TensorData::from([10.0, 20.0]), &device).require_grad();
    // Multiply data[0,1] by 10 and data[1,0] by 20.
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1], [1, 0]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Mul);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data[0,1] = 10 (multiplier), grad_data[1,0] = 20, others = 1.
    grad_data
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 10.0], [20.0, 1.0]]), false);
    // grad_values = data at [0,1]=2, [1,0]=3.
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([2.0, 3.0]), false);
}

#[test]
fn test_scatter_nd_mul_grad_only_data_tracked() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device).require_grad();
    let values: TestTensor<1> = TestTensor::from_data(TensorData::from([10.0]), &device);
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1]]), &device);

    let result: TestTensor<2> = data
        .clone()
        .scatter_nd(indices, values, IndexingUpdateOp::Mul);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();

    grad_data
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 10.0], [1.0, 1.0]]), false);
}

#[test]
fn test_scatter_nd_mul_grad_only_values_tracked() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
    let values: TestTensor<1> =
        TestTensor::from_data(TensorData::from([10.0]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1]]), &device);

    let result: TestTensor<2> = data.scatter_nd(indices, values.clone(), IndexingUpdateOp::Mul);
    let grads = result.sum().backward();

    let grad_values = values.grad(&grads).unwrap();

    grad_values
        .to_data()
        .assert_eq(&TensorData::from([2.0]), false);
}

#[test]
fn test_scatter_nd_assign_grad_k_equals_d() {
    // K=D: scalar-level assign, each index overwrites a single element
    let device = AutodiffDevice::new();
    let data: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device).require_grad();
    let values: TestTensor<1> =
        TestTensor::from_data(TensorData::from([10.0, 20.0]), &device).require_grad();
    // indices shape: [2, 2] => K=2=D
    // Overwrite data[0,1] and data[1,0]
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0, 1], [1, 0]]), &device);

    let result: TestTensor<2> =
        data.clone()
            .scatter_nd(indices, values.clone(), IndexingUpdateOp::Assign);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data: ones everywhere except the overwritten positions are zeroed
    // [0,1] and [1,0] were overwritten, so those get 0
    grad_data
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 0.0], [0.0, 1.0]]), false);
    // grad_values = gather_nd(ones, indices) = [1.0, 1.0]
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([1.0, 1.0]), false);
}
