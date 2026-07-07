//! Tests that fused kernels actually write in-place (`HandleOutput::Alias`) when they can.
//!
//! Aliasing produces identical values to allocating a fresh output, so value-based tests
//! cannot catch a regression — the whole mechanism silently died for over a year without a
//! single test failing. These tests assert on the alias counter from
//! [`burn_cubecl_fusion::inspect`].
//!
//! The counter is process-global, so assertions are deltas (`>=`): concurrent tests can
//! only inflate it. "Does NOT alias" is asserted through values instead (a wrong alias
//! corrupts the still-referenced input).

use super::*;
use burn_cubecl_fusion::inspect::inplace_alias_count;
use burn_fusion::inspect::{BlockKind, FusionInspector};
use burn_tensor::{TensorData, Tolerance};

fn assert_all_fused(reports: &[burn_fusion::inspect::FusionReport], what: &str) {
    let tables = reports
        .iter()
        .map(|report| report.format_table())
        .collect::<Vec<_>>()
        .join("\n\n");

    assert!(
        !reports.is_empty()
            && reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .all(|block| matches!(block.kind, BlockKind::Fused { .. })),
        "{what}: every block should be fused, otherwise the aliasing assertion is \
         meaningless\n\n{tables}",
    );
}

/// A consuming element-wise chain must reuse the input buffer for its output.
///
/// This covers the two launch-side regressions that killed aliasing: the `can_mut()`
/// reference miscount (container clone kept during planning) and the reference-layout
/// gate that rejected the first output of every block.
#[test]
fn consuming_elemwise_chain_writes_inplace() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let mut tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        // Warmup with the same (relative) graph: first execution may go through autotune,
        // where benchmark trials run on forked contexts that (correctly) never alias.
        tensor = tensor.add_scalar(0.0).mul_scalar(1.0);
        let _ = tensor.clone().into_data();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let before = inplace_alias_count();
        for _ in 0..4 {
            tensor = tensor.add_scalar(1.0).mul_scalar(2.0);
            // Sync so every iteration is its own execution (own launch plan).
            let _ = tensor.clone().into_data();
        }
        device.sync().unwrap();
        let after = inplace_alias_count();

        tensor.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[46.0, 62.0], [78.0, 94.0]]),
            Tolerance::default(),
        );
        assert_all_fused(&inspector.drain(), "consuming elemwise chain");
        assert!(
            after - before >= 4,
            "each consuming iteration should alias its input buffer \
             (got {} aliases over 4 iterations)",
            after - before,
        );
    });
}

/// The consumed input keeps its buffer even when another input is broadcast: the output
/// (same shape as the consumed input) must alias it, and values must stay correct. This
/// is the shape that regressed with an invalid vector-size declaration on the alias.
#[test]
fn broadcast_elemwise_writes_inplace_and_computes_correctly() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let bias = TestTensor::<2>::from_data([[10.0, 20.0, 30.0, 40.0]], &device);
        let mut tensor =
            TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device);
        tensor = tensor.add_scalar(0.0).mul_scalar(1.0);
        let _ = tensor.clone().into_data();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let before = inplace_alias_count();
        let out = (tensor + bias).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[22.0, 44.0, 66.0, 88.0], [30.0, 52.0, 74.0, 96.0]]),
            Tolerance::default(),
        );
        device.sync().unwrap();
        let after = inplace_alias_count();

        assert_all_fused(&inspector.drain(), "broadcast elemwise");
        assert!(
            after > before,
            "the broadcast chain should alias the consumed input buffer",
        );
    });
}

/// A tensor that stays alive (read-only use) must NOT be written in-place: its values
/// must survive the op. Asserted through values — the alias counter is process-global,
/// so "did not move" can't be checked under parallel tests, but a wrong alias would
/// corrupt the still-referenced input.
#[test]
fn read_only_input_is_not_written_inplace() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let mut tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        tensor = tensor.add_scalar(0.0).mul_scalar(1.0);
        let _ = tensor.clone().into_data();
        device.sync().unwrap();

        // `tensor` stays alive across the chain: its buffer must not be reused.
        let out = tensor.clone().add_scalar(1.0).mul_scalar(2.0);
        out.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[4.0, 6.0], [8.0, 10.0]]),
            Tolerance::default(),
        );
        device.sync().unwrap();

        // The original tensor must be untouched.
        tensor.into_data().assert_approx_eq::<FloatElem>(
            &TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            Tolerance::default(),
        );
    });
}

/// Repeated consuming updates (the KV-cache pattern) must alias on every iteration while
/// the cached execution plan is reused, and values must stay correct throughout.
#[test]
fn repeated_consuming_updates_alias_with_cached_plan() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let mut acc = TestTensor::<1>::zeros([32], &device);
        acc = acc.add_scalar(0.0).mul_scalar(1.0);
        let _ = acc.clone().into_data();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let before = inplace_alias_count();
        for step in 1..=3 {
            let values = TestTensor::<1>::full([32], step as f32, &device);
            acc = (acc + values).add_scalar(1.0);
            let _ = acc.clone().into_data();
        }
        device.sync().unwrap();
        let after = inplace_alias_count();

        acc.into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from([9.0; 32]), Tolerance::default());
        assert_all_fused(&inspector.drain(), "repeated consuming updates");
        assert!(
            after - before >= 3,
            "each accumulation step should alias (got {} aliases over 3 steps)",
            after - before,
        );
    });
}
