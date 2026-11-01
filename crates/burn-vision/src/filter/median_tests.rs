use super::{MedianBlurConfig, median};
use burn_core::tensor::{DType, Device, Int, Tensor, TensorData, kind::Ordered, ops::PadMode};
use burn_std::{bf16, f16};

// Keep the original tensor decomposition as an independent backend-dispatch reference.
fn reference<K: Ordered>(images: Tensor<4, K>, size: usize) -> Tensor<4, K> {
    let dims @ [batch, channels, height, width] = images.dims();
    let reach = size / 2;
    let padded = images.pad((reach, reach, reach, reach), PadMode::Edge);
    let windows = (0..size)
        .flat_map(|y| (0..size).map(move |x| (y, x)))
        .map(|(y, x)| {
            padded
                .clone()
                .slice([0..batch, 0..channels, y..y + height, x..x + width])
        })
        .collect();
    let middle = size * size / 2;
    Tensor::stack::<5>(windows, 0)
        .sort(0)
        .slice(middle..middle + 1)
        .reshape(dims)
}

#[track_caller]
fn assert_exact<K: Ordered>(actual: Tensor<4, K>, expected: Tensor<4, K>) {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.device(), expected.device());
    // TensorData equality compares bytes, including signed zeros and NaN payloads.
    assert_eq!(actual.into_data(), expected.into_data());
}

#[track_caller]
fn assert_matches_reference<K: Ordered>(images: Tensor<4, K>, sizes: &[usize]) {
    for &size in sizes {
        assert_exact(
            median(images.clone(), size),
            reference(images.clone(), size),
        );
    }
}

#[test]
fn median_replicates_edges_and_selects_upper_even_rank_per_plane() {
    let device = Device::flex();
    let values = (0..4)
        .flat_map(|plane| [1., 4., 3., 2.].map(|v| v + plane as f32 * 10.))
        .collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(TensorData::new(values, [2, 2, 2, 2]), &device);

    for (size, plane) in [
        (1, [1., 4., 3., 2.]),
        (2, [1., 4., 3., 3.]),
        (3, [2., 3., 3., 2.]),
    ] {
        let values = (0..4)
            .flat_map(|i| plane.map(|v| v + i as f32 * 10.))
            .collect::<Vec<_>>();
        let expected = Tensor::<4>::from_data(TensorData::new(values, [2, 2, 2, 2]), &device);
        assert_exact(median(images.clone(), size), expected);
    }
}

#[test]
fn median_handles_views_and_vector_tails() {
    let values = (0..2 * 2 * 39 * 9)
        .map(|i| ((i * 71 + i / 9) % 293) as f32 - 140.25)
        .collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(TensorData::new(values, [2, 2, 39, 9]), &Device::flex());
    let transposed = images.clone().transpose();
    let sliced = transposed.clone().slice([0..2, 0..2, 1..8, 2..37]);
    let flipped = images.flip([2, 3]).transpose();

    for view in [transposed, sliced, flipped] {
        assert_matches_reference(view, &[3, 7]);
    }
}

#[test]
fn median_preserves_float_total_order_and_payloads() {
    let bits = [
        0xffc0_0002,
        0xff80_0001,
        0xff80_0000,
        0xbf80_0000,
        0x8000_0001,
        0x8000_0000,
        0x0000_0000,
        0x0000_0001,
        0x3eaa_aaab,
        0x3f80_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fc0_0001,
        0x7fc0_0002,
        0x7fff_ffff,
    ];
    let values = (0..2 * 5 * 35)
        .map(|i| f32::from_bits(bits[(i * 7 + i / 35) % bits.len()]))
        .collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(TensorData::new(values, [1, 2, 5, 35]), &Device::flex());
    assert_matches_reference(images, &[3, 7]);
}

#[test]
fn median_handles_exact_rgb8_levels_and_rejects_approximate_levels() {
    let dims = [1, 2, 17, 19];
    let values = (0..2 * 17 * 19)
        .map(|i| ((i * 73 + i / 19 * 29) % 256) as f32 / 255.)
        .collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(TensorData::new(values, dims), &Device::flex());
    assert_matches_reference(images, &[4, 6, 7, 15]);

    // Invalid levels occur after the initial probe and dominate later neighborhoods.
    let level = 128.0_f32 / 255.;
    for (exact, inexact) in [(level, f32::from_bits(level.to_bits() + 1)), (0., -0.)] {
        let mut values = vec![inexact; 17 * 19];
        values[..64].fill(exact);
        let images =
            Tensor::<4>::from_data(TensorData::new(values, [1, 1, 17, 19]), &Device::flex());
        assert_matches_reference(images, &[7]);
    }
}

#[test]
fn median_preserves_constant_planes_with_oversized_windows() {
    let values = [0x8000_0000, 0x7fc0_0042, 0xffc0_0042, 0x7f80_0000]
        .into_iter()
        .flat_map(|bits| [f32::from_bits(bits); 3])
        .collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(TensorData::new(values, [2, 2, 1, 3]), &Device::flex());
    for size in [3, 7] {
        assert_exact(median(images.clone(), size), images.clone());
    }
}

#[test]
fn median_preserves_f64_precision_and_total_order() {
    let bits = [
        0xfff8_0000_0000_0002,
        0xfff0_0000_0000_0001,
        f64::NEG_INFINITY.to_bits(),
        (-1.0_f64).to_bits(),
        (-0.0_f64).to_bits(),
        0,
        1,
        1.0_f64.to_bits(),
        1.0_f64.to_bits() + 1,
        1.0_f64.to_bits() + 2,
        f64::MAX.to_bits(),
        f64::INFINITY.to_bits(),
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_0000_0002,
    ];
    let values = (0..2 * 5 * 35)
        .map(|i| f64::from_bits(bits[(i * 7 + i / 35) % bits.len()]))
        .collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(
        TensorData::new(values, [1, 2, 5, 35]),
        (&Device::flex(), DType::F64),
    );
    assert_matches_reference(images, &[3, 7]);
}

#[test]
fn median_matches_half_precision_nan_conversion_and_ordering() {
    macro_rules! check {
        ($ty:ty, $dtype:expr, $bits:expr) => {{
            let bits = $bits;
            let values = (0..2 * 5 * 35)
                .map(|i| <$ty>::from_bits(bits[(i * 7 + i / 35) % bits.len()]))
                .collect::<Vec<_>>();
            let images = Tensor::<4>::from_data(
                TensorData::new(values, [1, 2, 5, 35]),
                (&Device::flex(), $dtype),
            );
            assert_matches_reference(images.clone(), &[3, 7]);
            assert_matches_reference(images.autodiff(), &[3, 7]);

            // Flex sorting quiets signaling half NaNs, including constant and size-one windows.
            let values = vec![<$ty>::from_bits(bits[1]); 3];
            let images = Tensor::<4>::from_data(
                TensorData::new(values, [1, 1, 1, 3]),
                (&Device::flex(), $dtype),
            );
            assert_matches_reference(images.clone(), &[1, 3, 7]);
            assert_matches_reference(images.autodiff(), &[1, 3, 7]);
        }};
    }

    check!(
        f16,
        DType::F16,
        [
            0xfe02, 0xfc01, 0xfc00, 0xbc00, 0x8001, 0x8000, 0x0000, 0x0001, 0x3555, 0x3c00, 0x7bff,
            0x7c00, 0x7c01, 0x7e01, 0x7e02, 0x7fff,
        ]
    );
    check!(
        bf16,
        DType::BF16,
        [
            0xffc2, 0xff81, 0xff80, 0xbf80, 0x8001, 0x8000, 0x0000, 0x0001, 0x3eab, 0x3f80, 0x7f7f,
            0x7f80, 0x7f81, 0x7fc1, 0x7fc2, 0x7fff,
        ]
    );
}

#[test]
fn median_preserves_all_integer_dtypes() {
    macro_rules! check {
        ($ty:ty, $dtype:expr) => {{
            let levels: [$ty; 8] = [
                <$ty>::MIN,
                <$ty>::MIN + 1,
                <$ty>::MIN / 2,
                0,
                1,
                <$ty>::MAX / 2,
                <$ty>::MAX - 1,
                <$ty>::MAX,
            ];
            let values = (0..2 * 5 * 35)
                .map(|i| levels[(i * 5 + i / 35) % levels.len()])
                .collect::<Vec<_>>();
            let images = Tensor::<4, Int>::from_data(
                TensorData::new(values, [1, 2, 5, 35]),
                (&Device::flex(), $dtype),
            );
            assert_matches_reference(images, &[3, 7]);
        }};
    }

    check!(i8, DType::I8);
    check!(i16, DType::I16);
    check!(i32, DType::I32);
    check!(i64, DType::I64);
    check!(u8, DType::U8);
    check!(u16, DType::U16);
    check!(u32, DType::U32);
    check!(u64, DType::U64);
}

#[test]
fn median_keeps_wide_integer_values_exact() {
    let device = Device::flex();
    for values in [
        vec![i64::MIN + 1, i64::MIN + 4, i64::MIN + 3, i64::MIN + 2],
        vec![i64::MAX - 4, i64::MAX - 1, i64::MAX - 2, i64::MAX - 3],
    ] {
        let images = Tensor::<4, Int>::from_data(
            TensorData::new(values, [1, 1, 2, 2]),
            (&device, DType::I64),
        );
        assert_matches_reference(images, &[2, 3, 7]);
    }
    let values = vec![u64::MAX - 4, u64::MAX - 1, u64::MAX - 2, u64::MAX - 3];
    let images =
        Tensor::<4, Int>::from_data(TensorData::new(values, [1, 1, 2, 2]), (&device, DType::U64));
    assert_matches_reference(images, &[2, 3, 7]);
}

#[test]
fn median_filters_integer_views_and_all_byte_levels() {
    let values = (0..17 * 19)
        .map(|i| (i * 73 % 256) as u8)
        .collect::<Vec<_>>();
    let images = Tensor::<4, Int>::from_data(
        TensorData::new(values.clone(), [1, 1, 17, 19]),
        (&Device::flex(), DType::U8),
    );
    let view = images.clone().flip([2, 3]).transpose();
    assert_matches_reference(view, &[3, 6, 15]);
    assert_exact(
        MedianBlurConfig::new().init().apply(images.clone()),
        reference(images, 3),
    );

    let images = Tensor::<4, Int>::from_data(
        TensorData::new(
            values.into_iter().map(|value| value as i8).collect(),
            [1, 1, 17, 19],
        ),
        (&Device::flex(), DType::I8),
    );
    assert_matches_reference(images, &[6, 15]);
}

#[test]
fn median_handles_nonconstant_single_row_and_column_images() {
    let images = Tensor::<4, Int>::from_data(
        TensorData::new(vec![-4_i16, 9, -2, 7, 1], [1, 1, 1, 5]),
        (&Device::flex(), DType::I16),
    );
    assert_matches_reference(images.clone(), &[3, 4, 7]);
    assert_matches_reference(images.transpose(), &[3, 7]);
}

#[test]
fn median_preserves_autodiff_context_and_gradients() {
    let device = Device::flex().autodiff();
    let images = Tensor::<4>::from_data([[[[1., 4.], [3., 2.]]]], &device).require_grad();
    let output = median(images.clone(), 3);
    assert!(output.is_tracked());
    assert_exact(output.clone(), reference(images.clone(), 3));

    let grads = output.sum().backward();
    let expected = Tensor::<4>::from_data([[[[0., 0.], [2., 2.]]]], &Device::flex());
    assert_exact(images.grad(&grads).unwrap(), expected);
}

#[test]
fn median_matches_weighted_gradients_for_ties_and_replicated_borders() {
    let plane = [1_f32, 2., 2., 0., 0., 1., 2., 2., 2., 1., 0., 1.];
    let data = TensorData::new(
        (0..4)
            .flat_map(|channel| (0..12).map(move |i| plane[(i + channel) % 12]))
            .collect::<Vec<_>>(),
        [2, 2, 3, 4],
    );
    for checkpointing in [false, true] {
        let device = Device::flex().autodiff();
        let device = if checkpointing {
            device.gradient_checkpointing()
        } else {
            device
        };
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            for size in [2, 3, 7] {
                let actual_input = Tensor::<4>::from_data(data.clone(), (&device, dtype))
                    .transpose()
                    .flip([2])
                    .require_grad();
                let expected_input = Tensor::<4>::from_data(data.clone(), (&device, dtype))
                    .transpose()
                    .flip([2])
                    .require_grad();
                let actual = median(actual_input.clone(), size);
                let expected = reference(expected_input.clone(), size);
                assert!(actual.is_tracked());
                assert_exact(actual.clone(), expected.clone());
                let weights = Tensor::<4>::from_data(
                    TensorData::new(
                        (0..48).map(|i| (i % 3 + 1) as f32).collect::<Vec<_>>(),
                        [2, 2, 4, 3],
                    ),
                    (&Device::flex(), dtype),
                );
                let actual_grads = (actual.clone() * actual * weights.clone()).sum().backward();
                let expected_grads = (expected.clone() * expected * weights).sum().backward();
                assert_exact(
                    actual_input.grad(&actual_grads).unwrap(),
                    expected_input.grad(&expected_grads).unwrap(),
                );
            }
        }
    }
}

#[test]
fn median_preserves_untracked_autodiff_and_integer_contexts() {
    let device = Device::flex().autodiff().gradient_checkpointing();
    let images = Tensor::<4>::from_data([[[[1., 4.], [3., 2.]]]], &device);
    let output = median(images.clone(), 3);
    assert!(output.is_autodiff());
    assert!(!output.is_tracked());
    assert_eq!(
        output.gradient_checkpointing_strategy(),
        images.gradient_checkpointing_strategy(),
    );
    assert_exact(output, reference(images, 3));

    let images = Tensor::<4, Int>::from_data([[[[1, 4], [3, 2]]]], &device);
    assert_matches_reference(images, &[3, 7]);
}
