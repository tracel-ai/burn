//! A strided view read by an element-wise op that has not run yet keeps reading its source after
//! the source's handle is dropped.

use super::*;
use burn_tensor::{Device, TensorData, Tolerance};

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * seed + 0.7).sin() * 0.5)
        .collect()
}

/// Column 4 of a `[1, 16, 8]` source, read transposed as `[1, 1, 16]`, plus a `[1, 1, 16]`.
fn column_plus_row(drop_source: bool) -> TensorData {
    let device = Device::default();
    let source = TestTensor::<3>::from_data(TensorData::new(values(128, 3.0), [1, 16, 8]), &device);
    let row = TestTensor::<3>::from_data(TensorData::new(values(16, 1.0), [1, 1, 16]), &device);
    device.sync().unwrap();

    let out = source.clone().narrow(2, 4, 1).swap_dims(1, 2) + row;
    let kept = (!drop_source).then_some(source);
    let out = out.into_data();
    drop(kept);
    out
}

#[test]
fn a_transposed_view_reads_its_source_after_the_source_is_dropped() {
    let source = values(128, 3.0);
    let row = values(16, 1.0);
    let expected: Vec<f32> = (0..16).map(|c| source[c * 8 + 4] + row[c]).collect();
    let expected = TensorData::new(expected, [1, 1, 16]);

    column_plus_row(false).assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    column_plus_row(true).assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
