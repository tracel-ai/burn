#![allow(clippy::single_range_in_vec_init)]

use std::collections::HashMap;

use burn_core::tensor::TensorData;
use burn_vision::{ConnectedComponents, ConnectedStatsOptions, Connectivity};

mod common;
use common::*;

fn space_invader() -> [[i32; 14]; 9] {
    [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]
}

#[test]
fn should_support_8_connectivity() {
    let device = TestDevice::default().into();
    let tensor = TestTensorBool::<2>::from_data(space_invader(), &device);

    let output = tensor.connected_components(Connectivity::Eight);
    let expected = space_invader(); // All pixels are in the same group for 8-connected
    let expected = TestTensorInt::<2>::from(expected);

    normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);
}

#[test]
fn should_support_8_connectivity_with_stats() {
    let device = TestDevice::default().into();
    let tensor = TestTensorBool::<2>::from_data(space_invader(), &device);

    let (output, stats) =
        tensor.connected_components_with_stats(Connectivity::Eight, ConnectedStatsOptions::all());
    let expected = space_invader(); // All pixels are in the same group for 8-connected
    let expected = TestTensorInt::<2>::from(expected);

    let (area, left, top, right, bottom) = (
        stats.area.slice([1..2]).into_data(),
        stats.left.slice([1..2]).into_data(),
        stats.top.slice([1..2]).into_data(),
        stats.right.slice([1..2]).into_data(),
        stats.bottom.slice([1..2]).into_data(),
    );

    output.into_data().assert_eq(&expected.into_data(), false);

    area.assert_eq(&TensorData::from([58]), false);
    left.assert_eq(&TensorData::from([0]), false);
    top.assert_eq(&TensorData::from([0]), false);
    right.assert_eq(&TensorData::from([13]), false);
    bottom.assert_eq(&TensorData::from([8]), false);
    stats
        .max_label
        .into_data()
        .assert_eq(&TensorData::from([1]), false);
}

#[test]
fn should_support_4_connectivity() {
    let device = TestDevice::default().into();
    let tensor = TestTensorBool::<2>::from_data(space_invader(), &device);

    let output = tensor.connected_components(Connectivity::Four);
    let expected = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0],
        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        [4, 0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0, 5],
        [4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 5, 5],
        [4, 4, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 5, 5],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    ];
    let expected = TestTensorInt::<2>::from(expected);

    normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);
}

#[test]
fn should_support_4_connectivity_with_stats() {
    let device = TestDevice::default().into();
    let tensor = TestTensorBool::<2>::from_data(space_invader(), &device);

    let (output, stats) =
        tensor.connected_components_with_stats(Connectivity::Four, ConnectedStatsOptions::all());
    let expected = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0],
        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        [4, 0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0, 5],
        [4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 5, 5],
        [4, 4, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 5, 5],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    ];
    let expected = TestTensorInt::<2>::from(expected);

    // Slice off background and limit to compacted labels
    let (area, left, top, right, bottom) = (
        stats.area.slice([1..6]).into_data(),
        stats.left.slice([1..6]).into_data(),
        stats.top.slice([1..6]).into_data(),
        stats.right.slice([1..6]).into_data(),
        stats.bottom.slice([1..6]).into_data(),
    );

    output.into_data().assert_eq(&expected.into_data(), false);

    area.assert_eq(&TensorData::from([1, 1, 46, 5, 5]), false);
    left.assert_eq(&TensorData::from([3, 10, 1, 0, 12]), false);
    top.assert_eq(&TensorData::from([0, 0, 1, 5, 5]), false);
    right.assert_eq(&TensorData::from([3, 10, 12, 1, 13]), false);
    bottom.assert_eq(&TensorData::from([0, 0, 8, 7, 7]), false);
    stats
        .max_label
        .into_data()
        .assert_eq(&TensorData::from([5]), false);
}

/// Normalize labels to sequential since actual labels aren't required to be contiguous and
/// different algorithms can return different numbers even if correct
fn normalize_labels(mut labels: TensorData) -> TensorData {
    let mut next_label = 0;
    let mut mappings = HashMap::<i32, i32>::default();
    let data = labels.as_mut_slice::<i32>().unwrap();
    for label in data {
        if *label != 0 {
            let relabel = mappings.entry(*label).or_insert_with(|| {
                next_label += 1;
                next_label
            });
            *label = *relabel;
        }
    }
    labels
}

/// CPU fallback data stays compact while Fusion retains its existing image-sized metadata.
#[cfg(feature = "cpu")]
#[test]
fn cube_fallback_statistics_are_not_padded() {
    let device = burn_core::tensor::Device::cpu();
    for (shape, data, counts) in [
        ([0, 3], vec![], [1, 1]),
        ([3, 0], vec![], [1, 1]),
        ([1, 1], vec![true], [2, 2]),
        ([1, 1], vec![false], [1, 1]),
        ([2, 3], vec![true, false, true, false, true, false], [4, 2]),
        ([256, 256], vec![true; 256 * 256], [2, 2]),
    ] {
        for (connectivity, count) in [Connectivity::Four, Connectivity::Eight]
            .into_iter()
            .zip(counts)
        {
            for bits in 0..8 {
                let opts = ConnectedStatsOptions {
                    bounds_enabled: bits & 1 != 0,
                    max_label_enabled: bits & 2 != 0,
                    compact_labels: bits & 4 != 0,
                };
                let img =
                    TestTensorBool::<2>::from_data(TensorData::new(data.clone(), shape), &device);
                let (labels, stats) = img.connected_components_with_stats(connectivity, opts);
                assert_eq!(labels.dims(), shape);
                assert_eq!(*labels.into_data().shape(), shape.into());
                for stat in [stats.area, stats.left, stats.top, stats.right, stats.bottom] {
                    #[cfg(feature = "fusion")]
                    assert_eq!(stat.dims(), [data.len()]);
                    #[cfg(not(feature = "fusion"))]
                    {
                        assert_eq!(stat.dims(), [count]);
                        assert_eq!(*stat.into_data().shape(), [count].into());
                    }
                    // Fusion reconstructs tensors with its declared shape. Full-array reads
                    // and consumers retain main's mismatch; compact data is checked unfused.
                }
                stats
                    .max_label
                    .into_data()
                    .assert_eq(&TensorData::from([count as i32 - 1]), false);
            }
        }
    }
}
