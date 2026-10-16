//! Dropping an input before evaluating its derived tensors must not change their values.
//! However, dropping tensors does change whether the allocations can be reused,
//! which can expose subtle effects.

use super::*;
use burn_fusion::inspect::FusionInspector;

const VALUES: [f32; 12] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
];

#[test]
fn elementwise_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device = Default::default();
        let run = |drop_input| {
            let values: [f32; 32] = std::array::from_fn(|i| i as f32 + 1.0);
            let input = TestTensor::<2>::from_data(values.map(|v| [v]), &device).swap_dims(0, 1);
            let incremented = input.clone() + 1.0;
            if drop_input {
                drop(input);
            }
            (incremented.clone() + 1.0).into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn reduce_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device = Default::default();
        let run = |drop_input| {
            let values = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
            // Mixed layouts give scalar reference reads; bias permits wider computation.
            let doubled = values.clone() * 2.0;
            let input = doubled + values.swap_dims(0, 1);
            let bias = TestTensor::<2>::from_data([[1.0, 2.0]], &device);
            let biased = input.clone() + bias;
            if drop_input {
                drop(input);
            }
            let inspector = FusionInspector::install(burn_tensor::StreamId::current());
            // The epilogue selects reduction fusion without disabling other optimizers.
            let result = (biased.sum_dim(1) + 1.0).into_data();
            assert!(
                inspector
                    .drain()
                    .iter()
                    .flat_map(|report| &report.blocks)
                    .any(|block| block.fuser_name() == Some("Reduce")),
                "expected reduction fusion",
            );
            result
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn gather_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device = Default::default();
        let run = |drop_input| {
            let source = TestTensor::<2>::from_data([[3.0, 4.0]], &device);
            let indices = TestTensorInt::<2>::from_data([[0, 1]], &device);
            let input = TestTensor::<2>::from_data([[1.0], [2.0]], &device).swap_dims(0, 1);
            let incremented = input.clone() + 1.0;
            if drop_input {
                drop(input);
            }
            (incremented.clone() + source.gather(1, indices)).into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn select_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device = Default::default();
        let run = |drop_input| {
            let source = TestTensor::<2>::from_data([[3.0, 4.0]], &device);
            let indices = TestTensorInt::<1>::from_data([0, 1], &device);
            let input = TestTensor::<2>::from_data([[1.0], [2.0]], &device).swap_dims(0, 1);
            let incremented = input.clone() + 1.0;
            if drop_input {
                drop(input);
            }
            (incremented.clone() + source.select(1, indices)).into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn cat_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device = Default::default();
        let run = |drop_input| {
            let left = TestTensor::<2>::from_data([[3.0]], &device);
            let right = TestTensor::<2>::from_data([[4.0]], &device);
            let input = TestTensor::<2>::from_data([[1.0], [2.0]], &device).swap_dims(0, 1);
            let incremented = input.clone() + 1.0;
            if drop_input {
                drop(input);
            }
            (incremented.clone() + TestTensor::cat(vec![left, right], 1)).into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn dequantize_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device: burn_tensor::Device = Default::default();
        let run = |drop_input| {
            let source = TestTensor::<2>::from_data([VALUES], &device);
            let quant = source.quantize_dynamic(&device.settings().quantization.scheme);
            let input = TestTensor::<2>::from_data(VALUES.map(|v| [v]), &device).swap_dims(0, 1);
            let output = input.clone() + quant.dequantize();
            if drop_input {
                drop(input);
            }
            output.into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn dequantize_moved_packed_axis_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device: burn_tensor::Device = Default::default();
        let run = |drop_input| {
            let source = TestTensor::<3>::from_data([[[0.0; 4], [1.0; 4]]], &device);
            let quant = source
                .quantize_dynamic(&device.settings().quantization.scheme)
                .swap_dims(1, 2);
            let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
            let input = TestTensor::<3>::from_data(values.map(|row| row.map(|v| [v])), &device)
                .permute([2, 0, 1]);
            let incremented = input.clone() + 1.0;
            if drop_input {
                drop(input);
            }
            (incremented.clone() + quant.dequantize()).into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}

#[test]
fn dequantize_incremented_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device: burn_tensor::Device = Default::default();
        let run = |drop_input| {
            let source = TestTensor::<2>::from_data([VALUES], &device);
            let quant = source.quantize_dynamic(&device.settings().quantization.scheme);
            let input = TestTensor::<2>::from_data(VALUES.map(|v| [v]), &device).swap_dims(0, 1);
            let incremented = input.clone() + 1.0;
            if drop_input {
                drop(input);
            }
            (incremented.clone() + quant.dequantize()).into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_eq(&without_drop, true);
    });
}
