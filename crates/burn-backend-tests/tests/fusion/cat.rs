//! Tests that `Cat` fuses with following element-wise operations into a single kernel
//! and computes correct values, including along the vectorization (last) axis.

use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};
use burn_tensor::{TensorData, Tolerance};

/// `cat` on dim 0 followed by an element-wise op should collapse into a single
/// ElementWise fused kernel containing both operations.
#[test]
fn cat_then_elementwise_fuses_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<2>::from_data([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], &device);
        let b =
            TestTensor::<2>::from_data([[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]], &device);
        let dtype = a.dtype();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = TestTensor::cat(vec![a, b], 0).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [0.0, 2.0, 4.0, 6.0],
                [8.0, 10.0, 12.0, 14.0],
                [16.0, 18.0, 20.0, 22.0],
                [24.0, 26.0, 28.0, 30.0],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        let block = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .find(|block| block.operations.iter().any(matchers::is_cat()))
            .unwrap_or_else(|| panic!("no block containing cat found\n\n{tables}"));

        assert!(
            matches!(
                block.kind,
                BlockKind::Fused {
                    name: "ElementWise",
                    ..
                }
            ),
            "expected cat in an ElementWise fused block, got {:?}\n\n{tables}",
            block.kind,
        );
        assert!(
            block
                .operations
                .iter()
                .any(matchers::is_mul_scalar_float(dtype)),
            "MulScalar should share the fused block with Cat\n\n{tables}",
        );
    });
}

/// `cat` along the last (vectorization) axis with segments that aren't aligned to the
/// vectorization width: each element of a vector may come from a different input.
#[test]
fn cat_last_dim_with_unaligned_segments_fuses_and_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let b = TestTensor::<2>::from_data(
            [
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [60.0, 70.0, 80.0, 90.0, 100.0],
            ],
            &device,
        );
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = TestTensor::cat(vec![a, b], 1).add_scalar(1.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [2.0, 3.0, 4.0, 11.0, 21.0, 31.0, 41.0, 51.0],
                [5.0, 6.0, 7.0, 61.0, 71.0, 81.0, 91.0, 101.0],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_cat())),
            "cat should be part of a fused block\n\n{tables}",
        );
    });
}

/// `cat` of three inputs along a middle axis of rank-3 tensors.
#[test]
fn cat_three_inputs_middle_dim_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<3>::from_data([[[1.0, 2.0]], [[3.0, 4.0]]], &device);
        let b = TestTensor::<3>::from_data(
            [[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]],
            &device,
        );
        let c = TestTensor::<3>::from_data([[[13.0, 14.0]], [[15.0, 16.0]]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = TestTensor::cat(vec![a, b, c], 1).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [[2.0, 4.0], [10.0, 12.0], [14.0, 16.0], [26.0, 28.0]],
                [[6.0, 8.0], [18.0, 20.0], [22.0, 24.0], [30.0, 32.0]],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_cat())),
            "cat should be part of a fused block\n\n{tables}",
        );
    });
}

/// When a `cat` input is produced by a preceding element-wise op in the same stream
/// segment, the fuser must split (the input needs to be materialized) while still
/// producing correct values.
#[test]
fn cat_of_computed_input_splits_but_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = TestTensor::<2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
        device.sync().unwrap();

        // No sync between the mul and the cat: the fuser has to handle the split itself.
        let out = TestTensor::cat(vec![a.mul_scalar(10.0), b], 0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[10.0, 20.0], [30.0, 40.0], [5.0, 6.0], [7.0, 8.0]]),
            Tolerance::default(),
        );
        device.sync().unwrap();
    });
}

/// `cat` on int tensors goes through the `BaseInt` path and should fuse as well.
#[test]
fn cat_int_then_elementwise_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensorInt::<1>::from_data([1, 2, 3], &device);
        let b = TestTensorInt::<1>::from_data([4, 5], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = TestTensorInt::cat(vec![a, b], 0) + 10;
        out.into_data()
            .assert_eq(&TensorData::from([11, 12, 13, 14, 15]), false);
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_cat())),
            "int cat should be part of a fused block\n\n{tables}",
        );
    });
}

/// `cat` on bool tensors goes through the `BaseBool` path and should fuse with a following
/// element-wise op (`equal` is the base op available on bool).
#[test]
fn cat_bool_then_elementwise_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensorBool::<1>::from_data([true, false, true], &device);
        let b = TestTensorBool::<1>::from_data([false, true], &device);
        let c = TestTensorBool::<1>::from_data([true, true, false, false, true], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = TestTensorBool::cat(vec![a, b], 0).equal(c);
        out.into_data()
            .assert_eq(&TensorData::from([true, false, false, true, true]), false);
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_cat())),
            "bool cat should be part of a fused block\n\n{tables}",
        );
    });
}

/// `cat` of a non-contiguous (transposed) input: the fused kernel must follow the input's
/// actual strides. Concatenating on the last dim also exercises the per-element path.
#[test]
fn cat_transposed_input_last_dim_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        // Non-contiguous handle of shape [3, 2].
        let a = a.swap_dims(0, 1);
        let b = TestTensor::<2>::from_data([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = TestTensor::cat(vec![a, b], 1).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [2.0, 8.0, 20.0, 40.0],
                [4.0, 10.0, 60.0, 80.0],
                [6.0, 12.0, 100.0, 120.0],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_cat())),
            "cat of a transposed input should be part of a fused block\n\n{tables}",
        );
    });
}

/// The `cat` output is broadcast by the following element-wise op: the fused block's
/// reference shape is larger than the cat output along a non-cat axis, so cat input reads
/// must wrap around on that axis.
#[test]
fn cat_output_broadcast_by_elementwise_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device);
        let b = TestTensor::<2>::from_data([[10.0, 20.0, 30.0, 40.0, 50.0]], &device);
        let c = TestTensor::<2>::from_data([[100.0; 8], [200.0; 8]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        // cat([1, 3], [1, 5]) = [1, 8], broadcast against [2, 8].
        let out = TestTensor::cat(vec![a, b], 1) + c;
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [101.0, 102.0, 103.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                [201.0, 202.0, 203.0, 210.0, 220.0, 230.0, 240.0, 250.0],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_cat())),
            "broadcast cat should be part of a fused block\n\n{tables}",
        );
    });
}
