//! Joint dimension-collapsing for zipped element-wise iteration.
//!
//! The generic fallback for binary element-wise ops used to walk both
//! operands with a per-element [`StridedIter`](crate::strided_index::StridedIter)
//! odometer, paying non-inlined index bookkeeping on every element.
//! This module collapses the two layouts *jointly* into a minimum-rank
//! loop nest (the same technique as `ndarray::Zip` and the
//! single-layout `collapse_for_copy` in `tensor.rs`): adjacent dims are
//! merged only when the merge rule holds for both operands at once, so
//! the innermost collapsed dim becomes a long run whose stride pair is
//! loop-invariant. Callers then specialize the inner loop on that
//! stride pair (contiguous/contiguous, contiguous/broadcast, general
//! strided) and the compiler autovectorizes it.
//!
//! Two traversals are built on the nest: [`zip_map`], which allocates a
//! fresh output, and [`zip_apply_inplace`], which writes back into an
//! operand that [`ZipNest::lhs_is_dense_from_zero`] says it can reuse.
//! The in-place form matters because a broadcast op whose dense operand
//! is uniquely owned — every step of an eager chain like `a * b + c * d`
//! — would otherwise pay for a full-size output buffer and the page
//! faults from first-touching it, which at multi-megabyte sizes costs
//! several times the arithmetic itself.

use crate::layout::Layout;
use alloc::vec::Vec;

/// Max rank we're willing to handle without falling back to the
/// strided iterator. Burn tensors are capped at 8 dims in practice.
pub(crate) const ZIP_MAX_RANK: usize = 8;

/// A pair of layouts collapsed into a joint loop nest, stored in stack
/// arrays so the hot path never hits the allocator.
///
/// `shape` is the collapsed logical shape shared by both operands;
/// `lhs_strides`/`rhs_strides` are each operand's strides over that
/// collapsed shape. The last dim is the inner run consumed by the
/// specialized inner loops; the leading dims are walked by
/// [`ZipNest::for_each_run`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct ZipNest {
    pub ndim: usize,
    pub shape: [usize; ZIP_MAX_RANK],
    pub lhs_strides: [isize; ZIP_MAX_RANK],
    pub rhs_strides: [isize; ZIP_MAX_RANK],
    pub lhs_offset: usize,
    pub rhs_offset: usize,
}

impl ZipNest {
    /// Inner run length and the (lhs, rhs) stride pair of the innermost
    /// collapsed dim. Callers dispatch their inner-loop specialization
    /// on the stride pair. Must not be called when `ndim == 0`.
    #[inline]
    pub fn inner(&self) -> (usize, isize, isize) {
        let d = self.ndim - 1;
        (self.shape[d], self.lhs_strides[d], self.rhs_strides[d])
    }

    /// True when the lhs side walks storage densely in row-major order
    /// from index 0 — every collapsed stride equals the product of the
    /// sizes below it, and the start offset is 0.
    ///
    /// That is exactly the condition for the lhs buffer to double as the
    /// *destination* of an in-place op: run `k` of [`Self::for_each_run`]
    /// then reports an `lhs_base` equal to the output position it is
    /// producing, so each of the `numel` slots is written exactly once.
    /// Note this is weaker than [`Layout::is_contiguous`], which rejects
    /// a size-1 dim carrying a stride of 0 (what `expand`/`swap_dims`
    /// leave behind); the collapse squeezes those dims away first.
    pub fn lhs_is_dense_from_zero(&self) -> bool {
        if self.lhs_offset != 0 {
            return false;
        }
        let mut expected = 1isize;
        for d in (0..self.ndim).rev() {
            if self.lhs_strides[d] != expected {
                return false;
            }
            match expected.checked_mul(self.shape[d] as isize) {
                Some(next) => expected = next,
                None => return false,
            }
        }
        true
    }

    /// Call `f(lhs_base, rhs_base)` once per innermost run, in
    /// row-major output order. The bases are storage indices of the
    /// first element of the run; offsets advance by stride addition,
    /// never per-element index math. Must not be called when
    /// `ndim == 0` or when any dim is empty.
    pub fn for_each_run(&self, mut f: impl FnMut(usize, usize)) {
        debug_assert!(self.ndim >= 1);
        let outer = self.ndim - 1;
        let mut idx = [0usize; ZIP_MAX_RANK];
        let mut lhs_base = self.lhs_offset as isize;
        let mut rhs_base = self.rhs_offset as isize;
        loop {
            f(lhs_base as usize, rhs_base as usize);
            // Odometer over the outer dims, innermost-first. Each step
            // adds the dim's stride; a wrap subtracts the whole dim's
            // span (`shape * stride`, since the stride was added
            // `shape` times by then).
            let mut d = outer;
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                idx[d] += 1;
                lhs_base += self.lhs_strides[d];
                rhs_base += self.rhs_strides[d];
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
                lhs_base -= self.shape[d] as isize * self.lhs_strides[d];
                rhs_base -= self.shape[d] as isize * self.rhs_strides[d];
            }
        }
    }
}

/// Jointly collapse two same-shape layouts into the minimum-rank
/// equivalent loop nest:
///
/// 1. Squeeze size-1 dims (their stride never gets stepped past 0).
/// 2. Merge adjacent dims `(i, i+1)` when
///    `stride[i] == stride[i+1] * shape[i+1]` holds for *both*
///    operands, i.e. both walk the merged run linearly. Stride-0
///    (broadcast) dim pairs merge for free since `0 == 0 * n`.
///
/// Canonical example: `[2,S,N]` (strides `[S*N, N, 1]`) zipped with a
/// broadcast `[1,S,N]` (strides `[0, N, 1]`) collapses to `[2, S*N]`
/// with strides `[S*N, 1]` / `[0, 1]` — a contiguous SIMD-able inner
/// run of `S*N` elements repeated twice.
///
/// Returns `None` when the layouts can't be handled — rank above
/// [`ZIP_MAX_RANK`] or a negative stride (from `flip`; the merge rule
/// assumes non-negative strides) — so callers fall back to their
/// generic strided path.
pub(crate) fn collapse_for_zip(lhs: &Layout, rhs: &Layout) -> Option<ZipNest> {
    let shape = lhs.shape();
    let ndims = lhs.num_dims();
    debug_assert_eq!(
        &shape[..],
        &rhs.shape()[..],
        "collapse_for_zip: operands must be broadcast to the same shape"
    );
    if ndims > ZIP_MAX_RANK {
        return None;
    }
    let lhs_strides = lhs.strides();
    let rhs_strides = rhs.strides();
    if lhs_strides.iter().chain(rhs_strides).any(|&s| s < 0) {
        return None;
    }

    let mut nest = ZipNest {
        ndim: 0,
        shape: [0; ZIP_MAX_RANK],
        lhs_strides: [0; ZIP_MAX_RANK],
        rhs_strides: [0; ZIP_MAX_RANK],
        lhs_offset: lhs.start_offset(),
        rhs_offset: rhs.start_offset(),
    };

    // Single forward sweep, like `collapse_for_copy`: squeeze size-1
    // dims and merge whenever the current dim's `stride * size` equals
    // the previous output dim's stride for both operands. `checked_mul`
    // keeps a pathological overflowing layout from wrapping into an
    // incorrect merge decision.
    for d in 0..ndims {
        let size = shape[d];
        if size == 1 {
            continue;
        }
        let l_st = lhs_strides[d];
        let r_st = rhs_strides[d];
        let merge = nest.ndim > 0 && {
            let prev = nest.ndim - 1;
            (size as isize)
                .checked_mul(l_st)
                .is_some_and(|run| nest.lhs_strides[prev] == run)
                && (size as isize)
                    .checked_mul(r_st)
                    .is_some_and(|run| nest.rhs_strides[prev] == run)
        };
        if merge {
            nest.shape[nest.ndim - 1] *= size;
            nest.lhs_strides[nest.ndim - 1] = l_st;
            nest.rhs_strides[nest.ndim - 1] = r_st;
        } else {
            nest.shape[nest.ndim] = size;
            nest.lhs_strides[nest.ndim] = l_st;
            nest.rhs_strides[nest.ndim] = r_st;
            nest.ndim += 1;
        }
    }

    Some(nest)
}

/// Apply `op` over two zipped strided operands via a collapsed loop
/// nest, producing the results in row-major output order.
///
/// The inner loop is specialized on the collapsed innermost stride
/// pair: both contiguous, one contiguous + one broadcast (stride 0),
/// or general strided with loop-invariant strides. All variants are
/// monomorphized per call site so LLVM autovectorizes them.
///
/// Returns `None` when the layout pair can't be collapsed (negative
/// strides, rank too high); callers keep their `StridedIter` fallback
/// for that case.
pub(crate) fn zip_map<E, R, F>(
    lhs: &[E],
    lhs_layout: &Layout,
    rhs: &[E],
    rhs_layout: &Layout,
    op: F,
) -> Option<Vec<R>>
where
    E: Copy,
    F: Fn(E, E) -> R,
{
    let numel = lhs_layout.num_elements();
    if numel == 0 {
        return Some(Vec::new());
    }
    let nest = collapse_for_zip(lhs_layout, rhs_layout)?;

    let mut out: Vec<R> = Vec::with_capacity(numel);
    if nest.ndim == 0 {
        // All dims were size 1: a single element.
        out.push(op(lhs[nest.lhs_offset], rhs[nest.rhs_offset]));
        return Some(out);
    }

    let (len, l_st, r_st) = nest.inner();
    match (l_st, r_st) {
        (1, 1) => nest.for_each_run(|lb, rb| {
            out.extend(
                lhs[lb..lb + len]
                    .iter()
                    .zip(&rhs[rb..rb + len])
                    .map(|(&a, &b)| op(a, b)),
            );
        }),
        (1, 0) => nest.for_each_run(|lb, rb| {
            let b = rhs[rb];
            out.extend(lhs[lb..lb + len].iter().map(|&a| op(a, b)));
        }),
        (0, 1) => nest.for_each_run(|lb, rb| {
            let a = lhs[lb];
            out.extend(rhs[rb..rb + len].iter().map(|&b| op(a, b)));
        }),
        _ => nest.for_each_run(|lb, rb| {
            out.extend(
                (0..len).map(|i| op(lhs[lb + i * l_st as usize], rhs[rb + i * r_st as usize])),
            );
        }),
    }
    debug_assert_eq!(out.len(), numel);
    Some(out)
}

/// Apply `op` over a collapsed nest *in place*, writing the result back
/// into the lhs buffer instead of allocating an output.
///
/// The caller must have checked [`ZipNest::lhs_is_dense_from_zero`] (so
/// `dst` is written exactly once per element) and that the tensor owning
/// `dst` is uniquely referenced (so no other view observes the mutation,
/// and `dst` cannot alias `src`).
///
/// `op` receives `(dst_value, src_value)`. Callers that write into the
/// *right* operand pass a flipped closure and a nest built with the
/// operands swapped, so the original operand order is preserved.
///
/// Like [`zip_map`], the inner loop is specialized on the collapsed
/// innermost src stride — contiguous, broadcast-scalar, or general
/// strided — and monomorphized per call site so LLVM autovectorizes it.
pub(crate) fn zip_apply_inplace<E, F>(nest: &ZipNest, dst: &mut [E], src: &[E], op: F)
where
    E: Copy,
    F: Fn(E, E) -> E,
{
    debug_assert!(
        nest.lhs_is_dense_from_zero(),
        "zip_apply_inplace: destination must be dense from index 0"
    );
    if nest.ndim == 0 {
        // All dims were size 1: a single element.
        dst[0] = op(dst[0], src[nest.rhs_offset]);
        return;
    }
    if nest.shape[..nest.ndim].contains(&0) {
        // Empty tensor; `for_each_run` must not be entered.
        return;
    }

    let (len, l_st, r_st) = nest.inner();
    debug_assert_eq!(l_st, 1, "dense destination implies a contiguous inner run");
    match r_st {
        1 => nest.for_each_run(|lb, rb| {
            for (d, &s) in dst[lb..lb + len].iter_mut().zip(&src[rb..rb + len]) {
                *d = op(*d, s);
            }
        }),
        0 => nest.for_each_run(|lb, rb| {
            let s = src[rb];
            for d in dst[lb..lb + len].iter_mut() {
                *d = op(*d, s);
            }
        }),
        _ => nest.for_each_run(|lb, rb| {
            for (i, d) in dst[lb..lb + len].iter_mut().enumerate() {
                *d = op(*d, src[rb + i * r_st as usize]);
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strided_index::StridedIter;
    use alloc::vec;
    use burn_std::Shape;

    /// Reference result computed with the per-element StridedIter path.
    fn reference<E: Copy, R>(
        lhs: &[E],
        lhs_layout: &Layout,
        rhs: &[E],
        rhs_layout: &Layout,
        op: impl Fn(E, E) -> R,
    ) -> Vec<R> {
        StridedIter::new(lhs_layout)
            .zip(StridedIter::new(rhs_layout))
            .map(|(li, ri)| op(lhs[li], rhs[ri]))
            .collect()
    }

    fn broadcast_layout(shape: &[usize], full: &[usize]) -> Layout {
        // Mimic expand: stride 0 on broadcast dims, contiguous elsewhere.
        let contiguous = Layout::contiguous(Shape::from(shape.to_vec()));
        let mut strides = contiguous.strides().to_vec();
        for (d, (&s, &f)) in shape.iter().zip(full).enumerate() {
            if s == 1 && f != 1 {
                strides[d] = 0;
            }
        }
        Layout::new(Shape::from(full.to_vec()), strides, 0)
    }

    #[test]
    fn test_collapse_contiguous_pair_merges_fully() {
        let l = Layout::contiguous(Shape::from(vec![2, 3, 4]));
        let r = Layout::contiguous(Shape::from(vec![2, 3, 4]));
        let nest = collapse_for_zip(&l, &r).unwrap();
        assert_eq!(nest.ndim, 1);
        assert_eq!(nest.shape[0], 24);
        assert_eq!(nest.lhs_strides[0], 1);
        assert_eq!(nest.rhs_strides[0], 1);
    }

    #[test]
    fn test_collapse_leading_broadcast_merges_inner() {
        // [2,3,4] zip broadcast [1,3,4]: rhs strides [0,4,1] -> the two
        // inner dims merge on both sides, the leading dim can't.
        let l = Layout::contiguous(Shape::from(vec![2, 3, 4]));
        let r = broadcast_layout(&[1, 3, 4], &[2, 3, 4]);
        let nest = collapse_for_zip(&l, &r).unwrap();
        assert_eq!(nest.ndim, 2);
        assert_eq!(&nest.shape[..2], &[2, 12]);
        assert_eq!(&nest.lhs_strides[..2], &[12, 1]);
        assert_eq!(&nest.rhs_strides[..2], &[0, 1]);
    }

    #[test]
    fn test_collapse_rejects_negative_strides() {
        let l = Layout::contiguous(Shape::from(vec![2, 3])).flip(&[0]);
        let r = Layout::contiguous(Shape::from(vec![2, 3]));
        assert!(collapse_for_zip(&l, &r).is_none());
    }

    #[test]
    fn test_zip_map_matches_strided_iter_broadcast_shapes() {
        // The issue #5069 shapes, scaled down: every broadcast
        // orientation must match the per-element reference exactly.
        let s = 5;
        let n = 7;
        let full = [2usize, s, n];
        let dense: Vec<f32> = (0..2 * s * n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let cases: Vec<(Vec<usize>, usize)> = vec![
            (vec![1, s, 1], s),
            (vec![1, s, n], s * n),
            (vec![2, 1, 1], 2),
            (vec![1, 1, 1], 1),
            (vec![2, s, 1], 2 * s),
        ];
        let dense_layout = Layout::contiguous(Shape::from(full.to_vec()));
        for (bshape, belems) in cases {
            let bdata: Vec<f32> = (0..belems).map(|i| i as f32 - 3.0).collect();
            let blayout = broadcast_layout(&bshape, &full);
            // Broadcast on the rhs...
            let got = zip_map(&dense, &dense_layout, &bdata, &blayout, |a, b| a * b).unwrap();
            let want = reference(&dense, &dense_layout, &bdata, &blayout, |a, b| a * b);
            assert_eq!(got, want, "rhs-broadcast {bshape:?}");
            // ...and on the lhs (non-commutative op to catch swaps).
            let got = zip_map(&bdata, &blayout, &dense, &dense_layout, |a, b| a - b).unwrap();
            let want = reference(&bdata, &blayout, &dense, &dense_layout, |a, b| a - b);
            assert_eq!(got, want, "lhs-broadcast {bshape:?}");
        }
    }

    #[test]
    fn test_zip_map_general_strided_inner() {
        // Transposed lhs: collapsed inner stride pair is neither
        // contiguous nor broadcast, exercising the general arm.
        let data: Vec<i32> = (0..12).collect();
        let l = Layout::contiguous(Shape::from(vec![3, 4])).transpose(0, 1); // [4,3], strides [1,4]
        let r = Layout::contiguous(Shape::from(vec![4, 3]));
        let rdata: Vec<i32> = (100..112).collect();
        let got = zip_map(&data, &l, &rdata, &r, |a, b| a + b).unwrap();
        let want = reference(&data, &l, &rdata, &r, |a, b| a + b);
        assert_eq!(got, want);
    }

    #[test]
    fn test_zip_map_offset_views() {
        // Narrowed operands: non-zero start offsets must carry through.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let l = Layout::contiguous(Shape::from(vec![4, 6])).narrow(0, 1, 2); // offset 6
        let r = Layout::contiguous(Shape::from(vec![4, 6])).narrow(0, 2, 2); // offset 12
        let got = zip_map(&data, &l, &data, &r, |a, b| a + b).unwrap();
        let want = reference(&data, &l, &data, &r, |a, b| a + b);
        assert_eq!(got, want);
    }

    #[test]
    fn test_zip_map_single_element() {
        let l = Layout::contiguous(Shape::from(vec![1, 1]));
        let r = Layout::contiguous(Shape::from(vec![1, 1]));
        let got = zip_map(&[3.0f32], &l, &[4.0f32], &r, |a, b| a * b).unwrap();
        assert_eq!(got, vec![12.0]);
    }

    /// Run `zip_apply_inplace` over `dst`, given that `dst_layout` is
    /// the dense side. Returns `None` if the nest declines the pair.
    fn apply_inplace<E: Copy>(
        dst: &mut [E],
        dst_layout: &Layout,
        src: &[E],
        src_layout: &Layout,
        op: impl Fn(E, E) -> E,
    ) -> Option<()> {
        let nest = collapse_for_zip(dst_layout, src_layout)?;
        if !nest.lhs_is_dense_from_zero() {
            return None;
        }
        zip_apply_inplace(&nest, dst, src, op);
        Some(())
    }

    #[test]
    fn test_dense_from_zero_accepts_size_one_dim_with_stride_zero() {
        // What `expand`/`swap_dims` leave behind: a size-1 dim carrying
        // stride 0. `Layout::is_contiguous` rejects this, but the dim is
        // squeezed by the collapse, so the walk is still dense.
        let l = Layout::new(Shape::from(vec![2, 1, 4]), vec![4, 0, 1], 0);
        assert!(!l.is_contiguous());
        let r = broadcast_layout(&[1, 1, 4], &[2, 1, 4]);
        assert!(collapse_for_zip(&l, &r).unwrap().lhs_is_dense_from_zero());
    }

    #[test]
    fn test_dense_from_zero_rejects_broadcast_offset_and_transpose() {
        let full = [2usize, 3, 4];
        let dense = Layout::contiguous(Shape::from(full.to_vec()));
        // A broadcast destination would write some slots many times.
        let bcast = broadcast_layout(&[1, 3, 4], &full);
        assert!(
            !collapse_for_zip(&bcast, &dense)
                .unwrap()
                .lhs_is_dense_from_zero()
        );
        // A non-zero start offset means run 0 doesn't begin at slot 0.
        let offset = Layout::contiguous(Shape::from(vec![4, 6])).narrow(0, 1, 2);
        let other = Layout::contiguous(Shape::from(vec![2, 6]));
        assert!(
            !collapse_for_zip(&offset, &other)
                .unwrap()
                .lhs_is_dense_from_zero()
        );
        // A transposed destination is not row-major.
        let t = Layout::contiguous(Shape::from(vec![3, 4])).transpose(0, 1);
        let c = Layout::contiguous(Shape::from(vec![4, 3]));
        assert!(!collapse_for_zip(&t, &c).unwrap().lhs_is_dense_from_zero());
    }

    #[test]
    fn test_zip_apply_inplace_matches_zip_map_broadcast_shapes() {
        // Same shape matrix as the `zip_map` broadcast test: the
        // in-place traversal must produce byte-identical results to the
        // allocating one, in both operand orders.
        let s = 5;
        let n = 7;
        let full = [2usize, s, n];
        let dense: Vec<f32> = (0..2 * s * n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let dense_layout = Layout::contiguous(Shape::from(full.to_vec()));
        let cases: Vec<(Vec<usize>, usize)> = vec![
            (vec![1, s, 1], s),
            (vec![1, s, n], s * n),
            (vec![2, 1, 1], 2),
            (vec![1, 1, 1], 1),
            (vec![2, s, 1], 2 * s),
        ];
        for (bshape, belems) in cases {
            let bdata: Vec<f32> = (0..belems).map(|i| i as f32 - 3.0).collect();
            let blayout = broadcast_layout(&bshape, &full);

            // Dense operand on the left: `dense - broadcast`.
            let want = zip_map(&dense, &dense_layout, &bdata, &blayout, |a, b| a - b).unwrap();
            let mut got = dense.clone();
            apply_inplace(&mut got, &dense_layout, &bdata, &blayout, |d, s| d - s)
                .expect("dense lhs must be reusable");
            assert_eq!(got, want, "dense-lhs {bshape:?}");

            // Dense operand on the right: `broadcast - dense`, written
            // into the dense operand with the closure flipped, which is
            // what `binary_op_typed`'s swapped branch does.
            let want = zip_map(&bdata, &blayout, &dense, &dense_layout, |a, b| a - b).unwrap();
            let mut got = dense.clone();
            apply_inplace(&mut got, &dense_layout, &bdata, &blayout, |d, s| s - d)
                .expect("dense rhs must be reusable");
            assert_eq!(got, want, "dense-rhs {bshape:?}");
        }
    }

    #[test]
    fn test_zip_apply_inplace_general_strided_source() {
        // Transposed source: the collapsed inner src stride is neither
        // contiguous nor broadcast, exercising the general arm.
        let dst_layout = Layout::contiguous(Shape::from(vec![4, 3]));
        let src_layout = Layout::contiguous(Shape::from(vec![3, 4])).transpose(0, 1);
        let src: Vec<i32> = (0..12).collect();
        let dst: Vec<i32> = (100..112).collect();

        let want = zip_map(&dst, &dst_layout, &src, &src_layout, |a, b| a + b).unwrap();
        let mut got = dst.clone();
        apply_inplace(&mut got, &dst_layout, &src, &src_layout, |d, s| d + s).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn test_zip_apply_inplace_single_element_and_empty() {
        let l = Layout::contiguous(Shape::from(vec![1, 1]));
        let mut dst = [3.0f32];
        apply_inplace(&mut dst, &l, &[4.0f32], &l, |d, s| d * s).unwrap();
        assert_eq!(dst, [12.0]);

        // Empty tensors must not enter the run odometer.
        let e = Layout::contiguous(Shape::from(vec![0, 3]));
        apply_inplace::<f32>(&mut [], &e, &[], &e, |d, s| d + s).unwrap();
    }

    #[test]
    fn test_zip_apply_inplace_leaves_trailing_storage_untouched() {
        // A dense-from-zero view over a longer buffer: only the first
        // `numel` slots may be written.
        let mut data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let dst_layout = Layout::new(Shape::from(vec![2, 6]), vec![6, 1], 0);
        let src_layout = broadcast_layout(&[2, 1], &[2, 6]);
        apply_inplace(
            &mut data,
            &dst_layout,
            &[10.0, 20.0],
            &src_layout,
            |d, s| d + s,
        )
        .unwrap();

        let mut want: Vec<f32> = (0..24).map(|i| i as f32).collect();
        for (i, w) in want.iter_mut().enumerate().take(12) {
            *w += if i < 6 { 10.0 } else { 20.0 };
        }
        assert_eq!(data, want);
    }

    #[test]
    fn test_zip_map_empty() {
        let l = Layout::contiguous(Shape::from(vec![0, 3]));
        let r = Layout::contiguous(Shape::from(vec![0, 3]));
        let got = zip_map::<f32, f32, _>(&[], &l, &[], &r, |a, b| a + b).unwrap();
        assert!(got.is_empty());
    }
}
