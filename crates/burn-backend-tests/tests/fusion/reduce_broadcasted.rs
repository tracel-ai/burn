use super::*;
use burn_tensor::{TensorData, Tolerance};

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

    // Forces previous tensors to be materialized.
    device.sync().unwrap();

    let x = tensor + fused_on_read.clone();
    let x = x.sum_dim(1);

    let x = x + fused_on_write;

    // Broadcast
    let end = x + fused_on_read;
    let actual = end.into_data();
    let expected = TensorData::from([
        [56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
        [193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0],
        [330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 337.0],
        [467.0, 468.0, 469.0, 470.0, 471.0, 472.0, 473.0, 474.0],
    ]);
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_reduce_broadcasted_2() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();
    let fused_on_read = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();
    let fused_on_write = TestTensorInt::<1>::arange(16..48, &device)
        .reshape([4, 8])
        .float();
    // Second fuse on read
    let y = TestTensorInt::<1>::arange(32..64, &device)
        .reshape([4, 8])
        .float();

    // Forces previous tensors to be materialized.
    device.sync().unwrap();

    let x = tensor + fused_on_read.clone();
    let x = x.sum_dim(1);
    let x = x + fused_on_write;
    let x = x.mean_dim(1);

    let end = x + y;
    device.sync().unwrap();

    let actual = end.into_data();
    let expected = TensorData::from([
        [107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5],
        [251.5, 252.5, 253.5, 254.5, 255.5, 256.5, 257.5, 258.5],
        [395.5, 396.5, 397.5, 398.5, 399.5, 400.5, 401.5, 402.5],
        [539.5, 540.5, 541.5, 542.5, 543.5, 544.5, 545.5, 546.5],
    ]);
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_reduce_broadcasted_3() {
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
    device.sync().unwrap();

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
fn test_reduce_broadcasted_4_reused_partial() {
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
    device.sync().unwrap();

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

// ---------------------------------------------------------------------------
// Regression test [#5201](https://github.com/tracel-ai/burn/issues/5202):
// A masked softmax fused by `ReduceBroadcastedFuser` was producing wrong values on wgpu.
// Identical wrong value every run (e.g. row 0 = 59874.13 instead of 1),
#[test]
fn test_reduce_broadcasted_masked_softmax() {
    let device = Default::default();
    let [d0, q, k] = [6usize, 96, 96];
    let n = (d0 * q * k) as i64;

    let x = TestTensorInt::<1>::arange(0..n, &device)
        .reshape([d0, q, k])
        .float();
    let x = (x / (n as f32) - 0.5) * 20.0;

    let q_idx = TestTensorInt::<1>::arange(0..q as i64, &device).reshape([q, 1]);
    let k_idx = TestTensorInt::<1>::arange(0..k as i64, &device).reshape([1, k]);
    let mask = k_idx.greater(q_idx).reshape([1, q, k]);
    device.sync().unwrap();

    // Softmax over the last dim of the masked scores.
    let scores = x.clone().mask_fill(mask, -10000.0);
    let max = scores.clone().max_dim(2);
    let e = (scores - max).exp();
    let sum = e.clone().sum_dim(2);
    let actual = (e / sum).into_data();

    // Host reference: the same causal-masked softmax, row by row.
    let xs = x.into_data();
    let xs = xs.as_slice::<FloatElem>().unwrap();
    let mut expected = vec![0f32; d0 * q * k];
    for o in 0..(d0 * q) {
        let qi = o % q;
        let row: Vec<f32> = (0..k)
            .map(|c| if c > qi { -10000.0 } else { xs[o * k + c] })
            .collect();
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for c in 0..k {
            expected[o * k + c] = exps[c] / s;
        }
    }

    assert!(
        actual
            .as_slice::<FloatElem>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "fused masked softmax produced non-finite output at [6, 96, 96]",
    );
    actual.assert_approx_eq::<FloatElem>(
        &TensorData::new(expected, [d0, q, k]),
        Tolerance::permissive(),
    );
}
