use super::*;
use burn_tensor::{TensorData, Tolerance, backend::Backend};

#[test]
fn test_reduce_broadcasted_1() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();
    let fused_on_read = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();
    let fused_on_write = TestTensorInt::<1>::arange(0..4, &device)
        .reshape([4, 1])
        .float();
    // Second fuse on read
    let y = TestTensorInt::<1>::arange(32..64, &device)
        .reshape([4, 8])
        .float();

    // Forces previous tensors to be materialized.
    TestBackend::sync(&device);

    let x = tensor + fused_on_read.clone();
    let x = x.sum_dim(1);

    let x = x + fused_on_write;

    // Broadcast
    let x = x + fused_on_read;
    // Second reduce
    let x = x.mean_dim(1);

    let end = x + y;
    let actual = end.into_data();
    let expected = TensorData::from([
        [91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5],
        [236.5, 237.5, 238.5, 239.5, 240.5, 241.5, 242.5, 243.5],
        [381.5, 382.5, 383.5, 384.5, 385.5, 386.5, 387.5, 388.5],
        [526.5, 527.5, 528.5, 529.5, 530.5, 531.5, 532.5, 533.5],
    ]);
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_reduce_broadcasted_2_reused_partial() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();
    let fused_on_read = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();
    let fused_on_write = TestTensorInt::<1>::arange(0..4, &device)
        .reshape([4, 1])
        .float();
    let y = TestTensorInt::<1>::arange(32..64, &device)
        .reshape([4, 8])
        .float();

    // Forces previous tensors to be materialized.
    TestBackend::sync(&device);

    // In fusion we have to create a global buffer to keep the intermediate data for now.
    let x_previous = tensor + fused_on_read;
    let x = x_previous.clone().sum_dim(1);

    let x = x * fused_on_write;

    // Broadcast
    let x = x + x_previous;
    // Second reduce
    let x = x.mean_dim(1);

    // Second fuse on read
    let end = x + y;
    let actual = end.into_data();
    let expected = TensorData::from([
        [39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0],
        [247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0],
        [711.0, 712.0, 713.0, 714.0, 715.0, 716.0, 717.0, 718.0],
        [
            1431.0, 1432.0, 1433.0, 1434.0, 1435.0, 1436.0, 1437.0, 1438.0,
        ],
    ]);
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
