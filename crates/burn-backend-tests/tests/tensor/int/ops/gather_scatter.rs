use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn should_gather_1d_dim0_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_ints([5, 6, 7], &device);
    let indices = TestTensorInt::from_ints([1, 1, 0, 1, 2], &device);

    let output = tensor.gather(0, indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([6, 6, 5, 6, 7]), false);
}

#[test]
fn should_gather_indices_broadcasted() {
    let device = Default::default();

    let batch_size = 3;
    let fft_size = 4;
    let shape = [batch_size, fft_size, 2];
    let x = TestTensorInt::arange(
        0..shape.iter().product::<usize>() as i64,
        &Default::default(),
    )
    .reshape(shape);
    let idx = TestTensorInt::<1>::from_ints([0, 2, 1, 3], &device);

    let expected = TestTensorInt::<3>::from([
        [[0, 1], [4, 5], [2, 3], [6, 7]],
        [[8, 9], [12, 13], [10, 11], [14, 15]],
        [[16, 17], [20, 21], [18, 19], [22, 23]],
    ])
    .into_data();

    // Case 1: gather dim 2
    let perm = idx
        .clone()
        .reshape([1, 1, fft_size])
        .repeat_dim(0, batch_size)
        .repeat_dim(1, 2);

    let input = x.clone().permute([0, 2, 1]);
    let out = input.gather(2, perm).permute([0, 2, 1]);

    out.into_data().assert_eq(&expected, true);

    // Case 2: gather directly on dim 1
    let perm = idx.reshape([1, fft_size, 1]).repeat_dim(0, batch_size);
    let out2 = x.gather(1, perm.repeat_dim(2, 2));

    out2.into_data().assert_eq(&expected, true);
}

#[test]
fn should_scatter_add_1d_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_ints([0, 0, 0], &device);
    let values = TestTensorInt::from_ints([5, 4, 3], &device);
    let indices = TestTensorInt::from_ints([1, 0, 2], &device);

    let output = tensor.scatter(0, indices, values, IndexingUpdateOp::Add);

    output
        .into_data()
        .assert_eq(&TensorData::from([4, 5, 3]), false);
}
