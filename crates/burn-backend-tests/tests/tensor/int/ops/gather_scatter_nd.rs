use super::*;
use burn_tensor::TensorData;

#[test]
fn test_int_gather_nd_2d() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[10, 20, 30], [40, 50, 60]], &device);
    // indices shape: [2, 2] => K=2, M=2, DV=1
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 2]], &device);

    let output: TestTensorInt<1> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([20, 60]), false);
}

#[test]
fn test_int_scatter_nd_add() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [1, 2]], &device);
    let values = TestTensorInt::<1>::from_ints([10, 20], &device);

    let output = data.scatter_nd_add(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[11, 2, 3], [4, 5, 26]]),
        false,
    );
}

#[test]
fn test_int_scatter_nd_mul() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[2, 3], [4, 5]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 0]], &device);
    let values = TestTensorInt::<1>::from_ints([10, 3], &device);

    let output = data.scatter_nd_mul(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[2, 30], [12, 5]]),
        false,
    );
}
