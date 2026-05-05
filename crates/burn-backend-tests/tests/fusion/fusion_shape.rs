//! Tests that assert on *which* operations were fused together, not just output values.
//!
//! These tests install a [`FusionInspector`], run a sequence of tensor ops, and then
//! inspect the captured [`FusionReport`]s to check that the expected operations ended
//! up in the same (or separate) fused kernels.
//!
//! `FusionInspector::install` captures operations on a specific stream, so two tests running
//! in parallel will not interfere when inspecting different streams. Use `test_stream()`
//! (which returns a fresh unique [`StreamId`] per call) to isolate each test. Installing
//! two inspectors for the *same* stream will panic.

use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};
use burn_tensor::backend::Backend;
use serial_test::serial;

/// `a + b` followed by `exp` should collapse into a single element-wise fused kernel.
#[test]
fn elementwise_add_then_exp_fuses_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        // Materialize the inputs first so that the `Ones` init ops aren't rolled into the
        // fused block we're trying to observe.
        let a = TestTensor::<2>::ones([4, 4], &device);
        let b = TestTensor::<2>::ones([4, 4], &device);
        let dtype = a.dtype();
        TestBackend::sync(&device).unwrap();

        let inspector = FusionInspector::install(stream);
        let out = (a + b).exp();
        let _ = out.into_data();
        TestBackend::sync(&device).unwrap();

        let reports = inspector.drain();
        assert!(!reports.is_empty(), "expected at least one fusion report");

        let target = reports
            .iter()
            .find(|r| r.total_operations() == 2)
            .unwrap_or_else(|| panic!("no report with 2 ops; got {reports:#?}"));

        let block = target.assert_single_fused_block();
        assert_eq!(block.fuser_name(), Some("ElementWise"));
        assert!(
            block.ops_match(&[matchers::is_add_float(dtype), matchers::is_exp(dtype),]),
            "block ops did not match add + exp: {:#?}",
            block.operations,
        );
    });
}

/// Forcing a sync between two sub-expressions should materialize the intermediate and
/// prevent fusion across the boundary.
#[test]
fn sync_between_ops_splits_into_separate_kernels() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let inspector = FusionInspector::install(stream);

        let a = TestTensor::<2>::ones([4, 4], &device);
        let b = TestTensor::<2>::ones([4, 4], &device);
        let intermediate = a + b;
        let dtype = intermediate.dtype();

        // Force materialization of the intermediate.
        TestBackend::sync(&device).unwrap();

        let out = intermediate.exp();
        let _ = out.into_data();
        TestBackend::sync(&device).unwrap();

        let reports = inspector.drain();

        // Across all reports, we should have at least one fused add and one fused exp, and
        // they should never appear together in the same block.
        let add = matchers::is_add_float(dtype);
        let exp = matchers::is_exp(dtype);
        for report in &reports {
            for block in &report.blocks {
                let has_add = block.operations.iter().any(|op| add(op));
                let has_exp = block.operations.iter().any(|op| exp(op));
                assert!(
                    !(has_add && has_exp),
                    "add and exp should not share a fused block after sync: {:#?}",
                    block,
                );
            }
        }

        let saw_add = reports.iter().any(|r| {
            r.blocks
                .iter()
                .any(|b| b.operations.iter().any(|op| add(op)))
        });
        let saw_exp = reports.iter().any(|r| {
            r.blocks
                .iter()
                .any(|b| b.operations.iter().any(|op| exp(op)))
        });
        assert!(
            saw_add && saw_exp,
            "expected both ops to appear; got {reports:#?}"
        );
    });
}

/// Multiple chained element-wise ops should still fuse into a single kernel.
#[test]
fn chained_elementwise_ops_fuse_together() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let a = TestTensor::<2>::ones([8, 8], &device);
        let b = TestTensor::<2>::ones([8, 8], &device);
        let c = TestTensor::<2>::ones([8, 8], &device);
        TestBackend::sync(&device).unwrap();

        let inspector = FusionInspector::install(stream);
        // add → mul → exp, all element-wise on the same shape.
        let out = ((a + b) * c).exp();
        let _ = out.into_data();
        TestBackend::sync(&device).unwrap();

        let reports = inspector.drain();

        let target = reports
            .iter()
            .find(|r| r.total_operations() == 3)
            .unwrap_or_else(|| panic!("no report with 3 ops; got {reports:#?}"));

        let block = target.assert_single_fused_block();
        assert!(
            matches!(
                block.kind,
                BlockKind::Fused {
                    name: "ElementWise",
                    ..
                }
            ),
            "expected a single ElementWise fused block, got {:?}",
            block.kind,
        );
        assert_eq!(block.operations.len(), 3);
    });
}

/// A loop that materializes new tensors with `ones` and folds them into an ongoing
/// elementwise computation. Every op — Ones, MulScalar, Mul, Add across every
/// iteration — should collapse into a single fused ElementWise kernel.
///
/// If this test fails, the panic message includes the fusion table so the split
/// points are visible directly in the test output.
#[test]
fn elementwise_and_creation_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
    const REPETITIONS: usize = 4;
    let device = Default::default();

    // Materialize the base tensor so it doesn't land in the inspector's first report.
    let original = TestTensor::<2>::ones([8, 8], &device);
    TestBackend::sync(&device).unwrap();

    let inspector = FusionInspector::install(stream);

    let mut tmp = original.clone();
    for i in 0..REPETITIONS {
        let new = TestTensor::<2>::ones([8, 8], &device) * (i as f32);
        tmp = tmp.clone().mul(original.clone()) + new;
    }

    let _ = tmp.into_data();
    TestBackend::sync(&device).unwrap();

    let reports = inspector.drain();
    assert!(!reports.is_empty(), "expected at least one fusion report");

    // Per iteration: ones, mul-by-scalar, mul, add  => 4 ops. `Drop` ops (memory
    // bookkeeping) also land in the reports but aren't counted here.
    let expected_ops = REPETITIONS * 4;
    let is_drop = matchers::is_drop();

    let tables: String = reports
        .iter()
        .map(|r| r.format_table())
        .collect::<Vec<_>>()
        .join("\n\n");

    // Everything should land in a single report...
    assert_eq!(
        reports.len(),
        1,
        "expected 1 report, got {}\n\n{tables}",
        reports.len(),
    );
    let report = &reports[0];

    // ...as a single fused block...
    assert_eq!(
        report.blocks.len(),
        1,
        "expected 1 block, got {}\n\n{tables}",
        report.blocks.len(),
    );
    let block = &report.blocks[0];

    // ...produced by the ElementWise fuser...
    assert_eq!(
        block.fuser_name(),
        Some("ElementWise"),
        "block was not ElementWise\n\n{tables}",
    );

    // ...containing exactly the expected non-Drop ops.
    let non_drop_ops = block.operations.iter().filter(|op| !is_drop(op)).count();
    assert_eq!(
        non_drop_ops, expected_ops,
        "expected {expected_ops} non-Drop ops in the fused block, got {non_drop_ops}\n\n{tables}",
    );
    });
}
