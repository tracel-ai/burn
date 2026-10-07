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
                    assert_eq!(stat.dims(), [data.len() + 1]);
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

#[cfg(any(feature = "cuda", feature = "vulkan"))]
mod accelerated {
    use super::*;
    use burn_core::{
        backend::ops::BoolTensorOps,
        tensor::{BoolStore, DType},
    };
    use burn_cubecl::{CubeBackend, CubeDevice, ops::into_data_sync, tensor::CubeTensor};
    use burn_vision::BoolVisionOps;

    /// Exercise public dispatch and Fusion metadata with a label equal to the image size.
    fn assert_last_pixel_statistics(opts: ConnectedStatsOptions, label: i32) {
        let cube_device: burn_cubecl::CubeDevice = TestDevice::default().into();
        let client = cube_device.client();
        let props = &client.properties().hardware;
        if props.plane_size_min != 32 || props.plane_size_max != 32 {
            return;
        }
        let device = TestDevice::default().into();
        let mut pixels = vec![false; 128];
        pixels[127] = true;
        let tensor = TestTensorBool::<2>::from_data(TensorData::new(pixels, [4, 32]), &device);
        let (labels, stats) = tensor.connected_components_with_stats(Connectivity::Four, opts);
        let mut expected_labels = vec![0i32; 128];
        expected_labels[127] = label;
        labels
            .into_data()
            .assert_eq(&TensorData::new(expected_labels, [4, 32]), false);
        for (stat, expected) in [
            (
                stats.area,
                (opts != ConnectedStatsOptions::none()).then_some(1),
            ),
            (stats.left, opts.bounds_enabled.then_some(31)),
            (stats.top, opts.bounds_enabled.then_some(3)),
            (stats.right, opts.bounds_enabled.then_some(31)),
            (stats.bottom, opts.bounds_enabled.then_some(3)),
        ] {
            assert_eq!(stat.dims(), [129]);
            let data = stat.into_data();
            assert_eq!(data.shape().dims::<1>(), [129]);
            if let Some(expected) = expected {
                assert_eq!(data.iter::<i32>().nth(label as usize), Some(expected));
            }
        }
        if opts.max_label_enabled || opts.compact_labels {
            stats
                .max_label
                .into_data()
                .assert_eq(&TensorData::from([label]), false);
        }
    }

    #[test]
    fn gpu_disabled_statistics_include_the_last_pixel() {
        assert_last_pixel_statistics(ConnectedStatsOptions::none(), 128);
    }

    #[test]
    fn gpu_sparse_statistics_include_the_last_label() {
        assert_last_pixel_statistics(
            ConnectedStatsOptions {
                compact_labels: false,
                ..ConnectedStatsOptions::all()
            },
            128,
        );
    }

    #[test]
    fn gpu_compact_statistics_include_the_last_label() {
        assert_last_pixel_statistics(ConnectedStatsOptions::all(), 1);
    }

    fn fixed_planes() -> bool {
        let device: CubeDevice = TestDevice::default().into();
        let client = device.client();
        let props = &client.properties().hardware;
        props.plane_size_min == 32 && props.plane_size_max == 32
    }

    fn image(data: TensorData) -> CubeTensor {
        CubeBackend::bool_from_data(
            data.convert_dtype(DType::Bool(BoolStore::U8)),
            &TestDevice::default().into(),
        )
    }

    // The two runs meet diagonally between rows 3 and 4 (a strip boundary).
    fn diagonal_runs(cols: usize, boundary: usize) -> Vec<i32> {
        let mut pixels = vec![0; 5 * cols];
        pixels[3 * cols..3 * cols + boundary].fill(1);
        pixels[4 * cols + boundary..].fill(1);
        pixels
    }

    fn diagonal_components(boundary: usize) -> (TensorData, Vec<i32>) {
        let cols = boundary + 3;
        let (labels, stats) = CubeBackend::connected_components_with_stats(
            image(TensorData::new(diagonal_runs(cols, boundary), [5, cols])),
            Connectivity::Eight,
            ConnectedStatsOptions::all(),
            DType::I32.into(),
        );
        // Primitive capacity distinguishes acceleration from compact CPU fallback output.
        assert_eq!(stats.area.meta.shape().dims::<1>(), [5 * cols + 1]);
        let areas = into_data_sync(stats.area)
            .iter::<i32>()
            .filter(|&area| area != 0)
            .collect();
        (into_data_sync(labels), areas)
    }

    #[test]
    fn eight_connectivity_joins_diagonal_runs_at_warp_boundary() {
        if !fixed_planes() {
            return;
        }
        // The diagonal crosses from lane 31 to lane 0 of the next warp.
        let (labels, areas) = diagonal_components(32);
        labels.assert_eq(&TensorData::new(diagonal_runs(35, 32), [5, 35]), false);
        assert_eq!(areas, [35]);
    }

    #[test]
    fn eight_connectivity_joins_diagonal_runs_at_merge_overlap_start() {
        if !fixed_planes() {
            return;
        }
        // With 32 warps per workgroup, the next group starts at 31 * 32 = 992.
        let (labels, areas) = diagonal_components(992);
        labels.assert_eq(&TensorData::new(diagonal_runs(995, 992), [5, 995]), false);
        assert_eq!(areas, [995]);
    }

    #[test]
    fn eight_connectivity_joins_diagonal_runs_at_merge_overlap_end() {
        if !fixed_planes() {
            return;
        }
        // Column 1024 is past the overlapping warp, in the next workgroup.
        let (labels, areas) = diagonal_components(1024);
        labels.assert_eq(
            &TensorData::new(diagonal_runs(1027, 1024), [5, 1027]),
            false,
        );
        assert_eq!(areas, [1027]);
    }

    #[test]
    fn all_background_image_has_no_components() {
        if !fixed_planes() {
            return;
        }
        let (labels, stats) = CubeBackend::connected_components_with_stats(
            image(TensorData::new(vec![false; 5 * 33], [5, 33])),
            Connectivity::Four,
            ConnectedStatsOptions::all(),
            DType::I32.into(),
        );
        assert_eq!(stats.area.meta.shape().dims::<1>(), [166]);
        into_data_sync(labels).assert_eq(&TensorData::new(vec![0i32; 165], [5, 33]), false);
        into_data_sync(stats.area).assert_eq(&TensorData::new(vec![0i32; 166], [166]), false);
        into_data_sync(stats.max_label).assert_eq(&TensorData::from([0]), false);
    }

    #[test]
    fn compaction_without_bounds_or_max_label_remaps_labels_and_area() {
        if !fixed_planes() {
            return;
        }
        // Sparse labels 2 and 12 become 1 and 2, even with both optional analyses disabled.
        let (labels, stats) = CubeBackend::connected_components_with_stats(
            image(TensorData::from([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]])),
            Connectivity::Four,
            ConnectedStatsOptions {
                bounds_enabled: false,
                max_label_enabled: false,
                compact_labels: true,
            },
            DType::I32.into(),
        );
        assert_eq!(stats.area.meta.shape().dims::<1>(), [13]);
        into_data_sync(labels).assert_eq(
            &TensorData::from([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 2]]),
            false,
        );
        // Old sparse entries must be cleared; compaction still needs the component count.
        into_data_sync(stats.area).assert_eq(
            &TensorData::from([0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            false,
        );
        into_data_sync(stats.max_label).assert_eq(&TensorData::from([2]), false);
    }

    #[test]
    fn padded_input_excludes_foreground_padding() {
        if !fixed_planes() {
            return;
        }
        let mut img = image(TensorData::from([[1, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1]]));
        // The fourth column is foreground padding, excluded from the logical image.
        *img.meta = burn_std::Metadata::new([3, 3], img.meta.strides().to_owned());
        let (labels, stats) = CubeBackend::connected_components_with_stats(
            img,
            Connectivity::Four,
            ConnectedStatsOptions::all(),
            DType::I32.into(),
        );
        assert_eq!(stats.area.meta.shape().dims::<1>(), [10]);
        into_data_sync(labels)
            .assert_eq(&TensorData::from([[1, 0, 2], [1, 0, 0], [1, 1, 0]]), false);
    }

    #[test]
    fn transposed_input_preserves_components() {
        if !fixed_planes() {
            return;
        }
        let img = image(TensorData::from([[1, 0, 1], [1, 0, 0], [1, 1, 0]]));
        let (labels, stats) = CubeBackend::connected_components_with_stats(
            CubeBackend::bool_swap_dims(img, 0, 1),
            Connectivity::Four,
            ConnectedStatsOptions::all(),
            DType::I32.into(),
        );
        assert_eq!(stats.area.meta.shape().dims::<1>(), [10]);
        into_data_sync(labels)
            .assert_eq(&TensorData::from([[1, 1, 1], [0, 0, 1], [2, 0, 0]]), false);
    }

    #[test]
    fn other_dtypes_keep_cpu_fallback() {
        let img = image(TensorData::new(vec![true; 16], [4, 4]));
        let labels =
            CubeBackend::connected_components(img.clone(), Connectivity::Eight, DType::I64.into());
        assert_eq!(labels.dtype, DType::I64);
        into_data_sync(labels).assert_eq(&TensorData::new(vec![1i64; 16], [4, 4]), false);
        let (labels, stats) = CubeBackend::connected_components_with_stats(
            img,
            Connectivity::Eight,
            ConnectedStatsOptions::all(),
            DType::I64.into(),
        );
        assert_eq!(labels.dtype, DType::I64);
        assert_eq!(stats.area.dtype, DType::I64);
        assert_eq!(stats.area.meta.shape().dims::<1>(), [2]);
        let area: Vec<i64> = into_data_sync(stats.area).iter::<i64>().collect();
        assert_eq!(area[1], 16);
    }
}
