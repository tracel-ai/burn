//! Run separately with BURN_TEST_NAN_POLICY=native and propagate.
//! General extrema, scan, clamp, and gradient coverage lives in the tensor and
//! autodiff suites; these cases cover native index validity and fusion plumbing.

#![cfg(feature = "cube")]

use burn_std::config::nan_policy;

pub type FloatElem = f32;
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
use backend::*;

#[test]
fn indexed_max_mixed_nan() {
    let row = [1.0, f32::NAN, 4.0, f32::NAN];
    let input = TestTensor::<2>::from([row]);
    let (values, indices) = input.clone().max_dim_with_indices(1);
    let value = values.into_data().as_slice::<FloatElem>().unwrap()[0];
    let index = indices.into_data().as_slice::<IntElem>().unwrap()[0];
    let arg = input.argmax(1).into_data().as_slice::<IntElem>().unwrap()[0];

    assert!((0..4).contains(&index));
    assert!((0..4).contains(&arg));
    let selected = row[index as usize];
    assert!(value == selected || (value.is_nan() && selected.is_nan()));
    if nan_policy().propagates_nan() {
        assert_eq!(index, 1);
        assert_eq!(arg, 1);
    }
}

#[test]
fn indexed_max_all_nan() {
    let input = TestTensor::<2>::from([[f32::NAN; 4]]);
    let (values, indices) = input.clone().max_dim_with_indices(1);
    let value = values.into_data().as_slice::<FloatElem>().unwrap()[0];
    let index = indices.into_data().as_slice::<IntElem>().unwrap()[0];
    let arg = input.argmax(1).into_data().as_slice::<IntElem>().unwrap()[0];

    assert!(value.is_nan());
    assert!((0..4).contains(&index));
    assert!((0..4).contains(&arg));
    if nan_policy().propagates_nan() {
        assert_eq!(index, 0);
        assert_eq!(arg, 0);
    }
}

#[test]
fn indexed_min_mixed_nan() {
    let row = [1.0, f32::NAN, 4.0, f32::NAN];
    let input = TestTensor::<2>::from([row]);
    let (values, indices) = input.clone().min_dim_with_indices(1);
    let value = values.into_data().as_slice::<FloatElem>().unwrap()[0];
    let index = indices.into_data().as_slice::<IntElem>().unwrap()[0];
    let arg = input.argmin(1).into_data().as_slice::<IntElem>().unwrap()[0];

    assert!((0..4).contains(&index));
    assert!((0..4).contains(&arg));
    let selected = row[index as usize];
    assert!(value == selected || (value.is_nan() && selected.is_nan()));
    if nan_policy().propagates_nan() {
        assert_eq!(index, 1);
        assert_eq!(arg, 1);
    }
}

#[test]
fn indexed_min_all_nan() {
    let input = TestTensor::<2>::from([[f32::NAN; 4]]);
    let (values, indices) = input.clone().min_dim_with_indices(1);
    let value = values.into_data().as_slice::<FloatElem>().unwrap()[0];
    let index = indices.into_data().as_slice::<IntElem>().unwrap()[0];
    let arg = input.argmin(1).into_data().as_slice::<IntElem>().unwrap()[0];

    assert!(value.is_nan());
    assert!((0..4).contains(&index));
    assert!((0..4).contains(&arg));
    if nan_policy().propagates_nan() {
        assert_eq!(index, 0);
        assert_eq!(arg, 0);
    }
}

#[test]
fn extrema_with_surrounding_operations() {
    let input = TestTensor::<2>::from([[1.0, f32::NAN, 4.0], [1.0, 2.0, 3.0]]);
    let output = input.mul_scalar(2.0).max_dim(1).add_scalar(1.0);
    let data = output.into_data().convert::<f32>();
    let values = data.as_slice::<f32>().unwrap();

    assert_eq!(values[1], 7.0);
    if nan_policy().propagates_nan() {
        assert!(values[0].is_nan());
    }
}

// Check the actual fuser choice: equivalent eager results would not exercise
// ReduceBroadcasted's separate policy resolution at launch.
#[cfg(all(feature = "cube", feature = "fusion"))]
#[test]
fn broadcasted_extrema_policy() {
    use burn_fusion::inspect::FusionInspector;
    use burn_tensor::{Device, StreamId};
    let device = Device::default();
    // Only CubeCL devices report an identity and use this fuser.
    if device.identity().is_none() {
        return;
    }
    let stream = StreamId::allocate();
    stream.executes(|| {
        for maximum in [false, true] {
            let tensor = TestTensor::<2>::zeros([2, 4], &device);
            let input = TestTensor::<2>::from_data(
                [[1.0, f32::NAN, 3.0, 2.0], [1.0, 4.0, 3.0, 2.0]],
                &device,
            );
            let bias = TestTensor::<2>::zeros([2, 1], &device);
            device.sync().unwrap();
            let inspector = FusionInspector::install(stream);
            let x = tensor + input.clone();
            let reduced = if maximum { x.max_dim(1) } else { x.min_dim(1) };
            let output = (reduced + bias + input + 1.0).into_data().convert::<f32>();
            let values = output.as_slice::<f32>().unwrap();
            assert_eq!(
                values[4..],
                if maximum {
                    [6.0, 9.0, 8.0, 7.0]
                } else {
                    [3.0, 6.0, 5.0, 4.0]
                }
            );
            if nan_policy().propagates_nan() {
                assert!(values[..4].iter().all(|x| x.is_nan()));
            }
            let reports = inspector.drain();
            assert!(
                reports
                    .iter()
                    .flat_map(|report| report.fused_blocks())
                    .any(|block| block.fuser_name() == Some("ReduceBroadcasted")),
                "expected ReduceBroadcasted fusion, got {reports:#?}"
            );
        }
    });
}
