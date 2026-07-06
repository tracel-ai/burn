//! Tests that `empty` + `slice_assign` chains fuse together with surrounding element-wise ops.
//!
//! `empty` used to close the fused block (it wasn't a recognized fuse op), so a
//! `Tensor::empty(...)` followed by a few `slice_assign`s and some element-wise math would split
//! into a standalone allocation plus one or more fused kernels. These tests pin down that the whole
//! chain now collapses into a single fused kernel, and that the result is still correct.

use super::*;
use burn_fusion::inspect::{FusionInspector, matchers};

/// `Tensor::empty` → a few `slice_assign` → element-wise should all land in one fused kernel.
#[test]
fn empty_slice_assign_elemwise_fuse_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        // Materialize the assigned values first so their creation ops don't land in the report we
        // inspect.
        let v0 = TestTensor::<2>::ones([2, 8], &device);
        let v1 = TestTensor::<2>::ones([2, 8], &device) * 2.0;
        let v2 = TestTensor::<2>::ones([2, 8], &device) * 3.0;
        let dtype = v0.dtype();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);

        let base = TestTensor::<2>::empty([8, 8], &device);
        let base = base.slice_assign([0..2, 0..8], v0);
        let base = base.slice_assign([2..4, 0..8], v1);
        let base = base.slice_assign([4..6, 0..8], v2);
        let out = base.exp() + 1.0;

        let _ = out.into_data();
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables: String = reports
            .iter()
            .map(|r| r.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        // Empty + 3×slice_assign + exp + add-scalar = 6 ops, all in one fused block.
        let target = reports
            .iter()
            .find(|r| r.total_operations() == 6)
            .unwrap_or_else(|| panic!("no report with 6 ops; got:\n\n{tables}"));

        let block = target.assert_single_fused_block();
        assert_eq!(
            block.fuser_name(),
            Some("ElementWise"),
            "expected a single ElementWise block:\n\n{tables}",
        );

        let is_empty = matchers::is_empty();
        let is_slice_assign = matchers::is_slice_assign();
        let is_exp = matchers::is_exp(dtype);

        assert_eq!(
            block.operations.iter().filter(|op| is_empty(op)).count(),
            1,
            "expected the empty allocation to be fused into the block:\n\n{tables}",
        );
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|op| is_slice_assign(op))
                .count(),
            3,
            "expected all 3 slice_assigns in the block:\n\n{tables}",
        );
        assert!(
            block.operations.iter().any(|op| is_exp(op)),
            "expected the trailing exp in the same block:\n\n{tables}",
        );
    });
}

/// Same shape of computation, but the `slice_assign`s fully cover the tensor so every element is
/// defined — lets us assert the fused kernel computes the right values.
#[test]
fn empty_slice_assign_elemwise_is_correct() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let top = TestTensor::<2>::ones([2, 4], &device) * 10.0;
        let bottom = TestTensor::<2>::ones([2, 4], &device) * 20.0;

        let base = TestTensor::<2>::empty([4, 4], &device);
        let base = base.slice_assign([0..2, 0..4], top);
        let base = base.slice_assign([2..4, 0..4], bottom);
        let out = base + 1.0;

        // Rows 0..2 were assigned 10 then +1 -> 11; rows 2..4 assigned 20 then +1 -> 21.
        let expected = TestTensor::<2>::from_floats(
            [
                [11.0, 11.0, 11.0, 11.0],
                [11.0, 11.0, 11.0, 11.0],
                [21.0, 21.0, 21.0, 21.0],
                [21.0, 21.0, 21.0, 21.0],
            ],
            &device,
        );

        out.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), burn_tensor::Tolerance::default());
    });
}
