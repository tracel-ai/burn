//! Tests that assert on *which* operations were fused together, not just output values.
//!
//! These tests install a [`FusionSpy`], run a sequence of tensor ops, and then inspect
//! the captured [`FusionReport`]s to check that the expected operations ended up in the
//! same (or separate) fused kernels.
//!
//! Each test is `#[serial]` because the spy is a process-global sink — two tests running
//! in parallel would contaminate each other's reports.

use super::*;
use burn_fusion::spy::{BlockKind, FusionSpy, matchers};
use burn_tensor::{DType, backend::Backend};
use serial_test::serial;

/// `a + b` followed by `exp` should collapse into a single element-wise fused kernel.
#[test]
#[serial]
fn elementwise_add_then_exp_fuses_into_single_kernel() {
    let device = Default::default();

    // Materialize the inputs first so that the `Ones` init ops aren't rolled into the
    // fused block we're trying to observe.
    let a = TestTensor::<2>::ones([4, 4], &device);
    let b = TestTensor::<2>::ones([4, 4], &device);
    TestBackend::sync(&device).unwrap();

    let spy = FusionSpy::install();
    let out = (a + b).exp();
    let _ = out.into_data();
    TestBackend::sync(&device).unwrap();

    let reports = spy.drain();
    assert!(!reports.is_empty(), "expected at least one fusion report");

    let target = reports
        .iter()
        .find(|r| r.total_operations() == 2)
        .unwrap_or_else(|| panic!("no report with 2 ops; got {reports:#?}"));

    let block = target.assert_single_fused_block();
    assert_eq!(block.fuser_name(), Some("ElementWise"));
    assert!(
        block.ops_match(&[
            matchers::is_add_float(DType::F32),
            matchers::is_exp(DType::F32),
        ]),
        "block ops did not match add + exp: {:#?}",
        block.operations,
    );
}

/// Forcing a sync between two sub-expressions should materialize the intermediate and
/// prevent fusion across the boundary.
#[test]
#[serial]
fn sync_between_ops_splits_into_separate_kernels() {
    let device = Default::default();
    let spy = FusionSpy::install();

    let a = TestTensor::<2>::ones([4, 4], &device);
    let b = TestTensor::<2>::ones([4, 4], &device);
    let intermediate = a + b;

    // Force materialization of the intermediate.
    TestBackend::sync(&device).unwrap();

    let out = intermediate.exp();
    let _ = out.into_data();
    TestBackend::sync(&device).unwrap();

    let reports = spy.drain();

    // Across all reports, we should have at least one fused add and one fused exp, and
    // they should never appear together in the same block.
    let add = matchers::is_add_float(DType::F32);
    let exp = matchers::is_exp(DType::F32);
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
    assert!(saw_add && saw_exp, "expected both ops to appear; got {reports:#?}");
}

/// Multiple chained element-wise ops should still fuse into a single kernel.
#[test]
#[serial]
fn chained_elementwise_ops_fuse_together() {
    let device = Default::default();

    let a = TestTensor::<2>::ones([8, 8], &device);
    let b = TestTensor::<2>::ones([8, 8], &device);
    let c = TestTensor::<2>::ones([8, 8], &device);
    TestBackend::sync(&device).unwrap();

    let spy = FusionSpy::install();
    // add → mul → exp, all element-wise on the same shape.
    let out = ((a + b) * c).exp();
    let _ = out.into_data();
    TestBackend::sync(&device).unwrap();

    let reports = spy.drain();

    let target = reports
        .iter()
        .find(|r| r.total_operations() == 3)
        .unwrap_or_else(|| panic!("no report with 3 ops; got {reports:#?}"));

    let block = target.assert_single_fused_block();
    assert!(
        matches!(block.kind, BlockKind::Fused { name: "ElementWise", .. }),
        "expected a single ElementWise fused block, got {:?}",
        block.kind,
    );
    assert_eq!(block.operations.len(), 3);
}
