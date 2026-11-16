use super::*;
use burn_tensor::TensorData;

#[test]
fn test_masked_scatter_float_1d() {
    let tensor = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0, 0.0]);
    let mask = TestTensorBool::<1>::from([true, false, true, true, false]);
    let source = TestTensor::<1>::from([10.0, 20.0, 30.0]);

    let output = tensor.masked_scatter(mask, source);

    output
        .into_data()
        .assert_eq(&TensorData::from([10.0, 0.0, 20.0, 30.0, 0.0]), false);
}

#[test]
fn test_masked_scatter_float_3d_row_major() {
    let tensor = TestTensor::<3>::zeros([2, 2, 2], &Default::default());
    let mask = TestTensorBool::<3>::from([
        [[true, false], [false, true]],
        [[false, false], [true, false]],
    ]);
    let source = TestTensor::<1>::from([1.0, 2.0, 3.0]);

    let output = tensor.masked_scatter(mask, source);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [3.0, 0.0]]]),
        false,
    );
}
