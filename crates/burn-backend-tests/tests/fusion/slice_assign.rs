//! Tests that `SliceAssign` fuses with surrounding element-wise operations into a single
//! kernel and computes correct values, including regions straddling the vectorization axis.
//!
//! Fusion is gated on the base tensor being consumed (`ReadWrite`) and unit steps: stepped
//! assigns and assigns whose base tensor is still referenced elsewhere keep the standalone
//! kernel.

use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};
use burn_tensor::{TensorData, Tolerance, s};

fn assert_slice_assign_fused(reports: &[burn_fusion::inspect::FusionReport], what: &str) {
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
                && block.operations.iter().any(matchers::is_slice_assign())),
        "{what} should be part of a fused block\n\n{tables}",
    );
}

fn assert_slice_assign_not_fused(reports: &[burn_fusion::inspect::FusionReport], what: &str) {
    let tables = reports
        .iter()
        .map(|report| report.format_table())
        .collect::<Vec<_>>()
        .join("\n\n");

    assert!(
        !reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .any(|block| matches!(block.kind, BlockKind::Fused { .. })
                && block.operations.iter().any(matchers::is_slice_assign())),
        "{what} should NOT be part of a fused block\n\n{tables}",
    );
}

/// `slice_assign` followed by an element-wise op should collapse into a single ElementWise
/// fused kernel containing both operations.
#[test]
fn slice_assign_then_elementwise_fuses_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor =
            TestTensor::<2>::from_data([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], &device);
        let value = TestTensor::<2>::from_data([[10.0, 20.0]], &device);
        let dtype = tensor.dtype();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_assign([1..2, 1..3], value).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[0.0, 2.0, 4.0, 6.0], [8.0, 20.0, 40.0, 14.0]]),
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
            .find(|block| block.operations.iter().any(matchers::is_slice_assign()))
            .unwrap_or_else(|| panic!("no block containing slice_assign found\n\n{tables}"));

        assert!(
            matches!(
                block.kind,
                BlockKind::Fused {
                    name: "ElementWise",
                    ..
                }
            ),
            "expected slice_assign in an ElementWise fused block, got {:?}\n\n{tables}",
            block.kind,
        );
        assert!(
            block
                .operations
                .iter()
                .any(matchers::is_mul_scalar_float(dtype)),
            "MulScalar should share the fused block with SliceAssign\n\n{tables}",
        );
    });
}

/// Interior region on both axes of a rank-3 tensor, with the trailing dim given as a full
/// range: only the starts matter, the extent comes from the value shape.
#[test]
fn slice_assign_interior_offsets_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<3>::zeros([2, 3, 2], &device);
        let value = TestTensor::<3>::from_data([[[1.0, 2.0], [3.0, 4.0]]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor
            .slice_assign(s![1.., 1..3, ..], value)
            .add_scalar(1.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "interior slice_assign");
    });
}

/// Region along the last (vectorization) axis whose boundaries aren't aligned to the
/// vectorization width: lanes of a single vector must pick different sources.
#[test]
fn slice_assign_last_dim_unaligned_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<2>::from_data(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            &device,
        );
        let value = TestTensor::<2>::from_data(
            [
                [100.0, 200.0, 300.0, 400.0, 500.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0],
            ],
            &device,
        );
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_assign(s![.., 1..6], value).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 12.0, 14.0],
                [16.0, -2.0, -4.0, -6.0, -8.0, -10.0, 28.0, 30.0],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "last-dim unaligned slice_assign");
    });
}

/// Stepped assigns keep the standalone kernel: correct values, no fused block.
#[test]
fn slice_assign_stepped_falls_back_but_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<1>::zeros([8], &device);
        let value = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_assign(s![0..8;2], value).add_scalar(1.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_not_fused(&inspector.drain(), "stepped slice_assign");
    });
}

/// When the base tensor is still referenced elsewhere (`ReadOnly` use), the fused path can't
/// reuse its buffer, so the standalone kernel (lazy copy + region write) is kept.
#[test]
fn slice_assign_read_only_tensor_falls_back_but_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
        let value = TestTensor::<1>::from_data([10.0, 20.0], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        // `tensor` stays alive across the slice_assign, so the op sees a read-only use.
        let assigned = tensor.clone().slice_assign([1..3], value);
        let out = assigned + tensor;
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([2.0, 12.0, 23.0, 8.0]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_not_fused(&inspector.drain(), "read-only slice_assign");
    });
}

/// `slice_fill` lowers to `slice_assign` with an expanded (stride-0) value tensor.
#[test]
fn slice_fill_fuses_and_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<2>::zeros([2, 4], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_fill(s![.., 1..3], 5.0).add_scalar(1.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[1.0, 6.0, 6.0, 1.0], [1.0, 6.0, 6.0, 1.0]]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "slice_fill");
    });
}

/// `slice_assign` on int tensors goes through the `BaseInt` path and should fuse as well.
#[test]
fn slice_assign_int_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensorInt::<1>::from_data([1, 2, 3, 4, 5], &device);
        let value = TestTensorInt::<1>::from_data([30, 40], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_assign([2..4], value) + 10;
        out.into_data()
            .assert_eq(&TensorData::from([11, 12, 40, 50, 15]), false);
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "int slice_assign");
    });
}

/// `slice_assign` on bool tensors goes through the `BaseBool` path and should fuse with a
/// following element-wise op (`equal` is the base op available on bool).
#[test]
fn slice_assign_bool_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensorBool::<1>::from_data([true, false, true, false], &device);
        let value = TestTensorBool::<1>::from_data([true, true], &device);
        let rhs = TestTensorBool::<1>::from_data([true, true, false, false], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_assign([1..3], value).equal(rhs);
        out.into_data()
            .assert_eq(&TensorData::from([true, true, false, true]), false);
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "bool slice_assign");
    });
}

/// KV-cache pattern: the same relative graph runs twice with different offsets (rebound at
/// launch from the range bindings), so the cached fused plan must be reused correctly.
#[test]
fn slice_assign_kv_cache_pattern_reuses_plan_with_different_offsets() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let mut cache = TestTensor::<2>::zeros([2, 6], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        for step in 0..2 {
            let fill = (step + 1) as f32 * 10.0;
            let value = TestTensor::<2>::from_data([[fill, fill + 1.0]; 2], &device);
            let start = step * 2;
            // The add keeps the fused block non-trivial, mirroring real post-processing.
            cache = cache
                .slice_assign(s![.., start..start + 2], value)
                .add_scalar(1.0);
        }
        cache.clone().into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [12.0, 13.0, 21.0, 22.0, 2.0, 2.0],
                [12.0, 13.0, 21.0, 22.0, 2.0, 2.0],
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

        let fused_count = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .filter(|block| {
                matches!(block.kind, BlockKind::Fused { .. })
                    && block.operations.iter().any(matchers::is_slice_assign())
            })
            .count();
        assert!(
            fused_count == 2,
            "both slice_assigns should run fused (got {fused_count})\n\n{tables}",
        );
    });
}

/// Non-contiguous (transposed) value input: the fused kernel must follow the value's actual
/// strides.
#[test]
fn slice_assign_transposed_value_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<2>::zeros([3, 4], &device);
        let value = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        // Non-contiguous handle of shape [2, 2].
        let value = value.swap_dims(0, 1);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = tensor.slice_assign(s![1..3, 2..4], value).add_scalar(1.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 4.0],
                [1.0, 1.0, 3.0, 5.0],
            ]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "transposed-value slice_assign");
    });
}

/// The `slice_assign` output is broadcast by the following element-wise op: the fused block's
/// reference shape is larger than the output along the leading axis, so reads must wrap.
#[test]
fn slice_assign_output_broadcast_by_elementwise_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device);
        let value = TestTensor::<2>::from_data([[20.0, 30.0]], &device);
        let rhs = TestTensor::<2>::from_data([[100.0; 4], [200.0; 4]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        // slice_assign([1, 4]) broadcast against [2, 4].
        let out = tensor.slice_assign([0..1, 1..3], value) + rhs;
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[101.0, 120.0, 130.0, 104.0], [201.0, 220.0, 230.0, 204.0]]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        assert_slice_assign_fused(&inspector.drain(), "broadcast slice_assign");
    });
}
