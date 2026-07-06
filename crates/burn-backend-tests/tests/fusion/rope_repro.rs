//! Regression test for a fusion search block-partitioning bug exposed by RoPE's `rotate_half`.
//!
//! `rotate_half` builds `[-x2, x1]` by writing two halves into a fresh `zeros` buffer with
//! `slice_assign`, then the caller multiplies by `sin`/`cos` and adds. The `zeros` (full width) and
//! the `neg` of `x2` (half width) are adjacent in the op stream, and the first `slice_assign`
//! references both. The search used to couple the `zeros` block to the differently-shaped `neg`
//! block through the slice-assign *value*; the shape-incompatible pair could not co-fuse, so `zeros`
//! was materialized in a separate un-fused block and the slice-assign chain read it as a global
//! input instead of fusing it. A `slice_assign` value is always a global (indexed) read, so it must
//! not establish block connectivity — with that fixed, `zeros` fuses with the slice-assign +
//! element-wise chain into one kernel.

use super::*;
use burn_fusion::inspect::{FusionInspector, matchers};
use core::ops::Range;

fn rotate_half(x: TestTensor<2>) -> TestTensor<2> {
    let device = x.device();
    let dims = x.dims();
    let d = dims[1];
    let half = d / 2;

    let x1 = x.clone().narrow(1, 0, half);
    let x2 = x.narrow(1, half, half);
    let x2 = x2.neg();

    let lower: [Range<usize>; 2] = [0..dims[0], 0..half];
    let upper: [Range<usize>; 2] = [0..dims[0], half..d];

    TestTensor::<2>::zeros(dims, &device)
        .slice_assign(lower, x2)
        .slice_assign(upper, x1)
}

/// The `zeros` + both `slice_assign`s + the `sin`/`cos` element-wise math should land in one fused
/// kernel; only the half-width `neg`/slices stay separate (they feed it as global reads).
///
/// Regression: the search used to materialize `zeros` in a separate un-fused block. `zeros` (no
/// inputs) is forced into its own block the moment it is registered — before its consumer is seen —
/// and the first `slice_assign` then references both that `zeros` block and the differently-shaped
/// `neg` block (through its `value` operand). The merge of those two producer blocks fails on the
/// shape mismatch; the stream now *defers* the `zeros` block (which can still fuse with the
/// slice-assign) instead of flushing it, so it re-enters the next round and fuses on-write.
#[test]
fn rope_rotate_half_zeros_fuses_with_slice_assign_chain() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let seq = 8;
        let d = 8;
        // Materialize inputs so their creation ops don't land in the inspected reports.
        let x = TestTensor::<2>::ones([seq, d], &device);
        let sin = TestTensor::<2>::ones([seq, d], &device) * 0.5;
        let cos = TestTensor::<2>::ones([seq, d], &device) * 0.25;
        let dtype = x.dtype();
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);

        let out = rotate_half(x.clone()) * sin + x * cos;
        let _ = out.into_data();
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables: String = reports
            .iter()
            .map(|r| r.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        let is_zeros = matchers::is_zeros();
        let is_slice_assign = matchers::is_slice_assign();

        // Find the fused block that holds the slice-assign chain.
        let fused = reports
            .iter()
            .flat_map(|r| r.fused_blocks())
            .find(|b| b.operations.iter().filter(|op| is_slice_assign(op)).count() == 2)
            .unwrap_or_else(|| {
                panic!("no fused block with both slice_assigns; got:\n\n{tables}")
            });

        // The zeros allocation must be fused into that same block, not left in a separate
        // (un-fused) block feeding it as a global input.
        assert_eq!(
            fused.operations.iter().filter(|op| is_zeros(op)).count(),
            1,
            "zeros should fuse with the slice_assign chain, not sit in a separate block:\n\n{tables}",
        );

        // And so should the trailing sin/cos element-wise math (2 muls + 1 add).
        let is_mul = matchers::is_mul_float(dtype);
        let is_add = matchers::is_add_float(dtype);
        assert_eq!(
            fused.operations.iter().filter(|op| is_mul(op)).count(),
            2,
            "both muls should be in the fused block:\n\n{tables}",
        );
        assert!(
            fused.operations.iter().any(|op| is_add(op)),
            "the add should be in the fused block:\n\n{tables}",
        );
    });
}

/// End-to-end value check for the same computation, so the fused kernel is proven correct.
#[test]
fn rope_rotate_half_is_correct() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        // x = [[1, 2, 3, 4]] so rotate_half(x) = [-x2, x1] = [-3, -4, 1, 2].
        let x = TestTensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let sin = TestTensor::<2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let cos = TestTensor::<2>::from_floats([[10.0, 10.0, 10.0, 10.0]], &device);

        // rotate_half(x)*sin + x*cos = [-3, -4, 1, 2] + [10, 20, 30, 40] = [7, 16, 31, 42].
        let out = rotate_half(x.clone()) * sin + x * cos;

        let expected = TestTensor::<2>::from_floats([[7.0, 16.0, 31.0, 42.0]], &device);
        out.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), burn_tensor::Tolerance::default());
    });
}
