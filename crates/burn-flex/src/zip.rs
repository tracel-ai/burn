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
    pub fn lhs_is_dense(&self) -> bool {
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

    pub fn lhs_is_dense_from_zero(&self) -> bool {
        self.lhs_offset == 0 && self.lhs_is_dense()
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
            // resets inner counters and adds the single delta to the
            // bases, so base addition is O(1) amortized per run.
            let mut d = outer;
            while d > 0 {
                d -= 1;
                idx[d] += 1;
                lhs_base += self.lhs_strides[d];
                rhs_base += self.rhs_strides[d];
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
                lhs_base -= (self.shape[d] as isize) * self.lhs_strides[d];
                rhs_base -= (self.shape[d] as isize) * self.rhs_strides[d];
            }
            if d == 0 && idx[0] == 0 {
                // Every dim rolled over back to 0; iteration is done.
                break;
            }
        }
    }
}

/// Jointly collapse two same-shape layouts into the minimum-rank
/// equivalent loop nest.
///
/// Returns `None` if `ndims > ZIP_MAX_RANK` or if any stride is negative
/// (flipped axes can't merge adjacent dims by scalar multiplication).
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
    let l_strides = lhs.strides();
    let r_strides = rhs.strides();
    if l_strides.iter().chain(r_strides).any(|&s| s < 0) {
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

    for d in 0..ndims {
        let size = shape[d];
        if size == 1 {
            // Size 1 dims don't advance indices; omit them.
            continue;
        }
        let l_st = l_strides[d];
        let r_st = r_strides[d];
        // Adjacent dims (prev, curr) can merge if stepping through
        // curr by its total length `size * curr_stride` covers exactly
        // the step of prev.
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
pub(crate) fn zip_map<L, R, Out, F>(
    lhs: &[L],
    lhs_layout: &Layout,
    rhs: &[R],
    rhs_layout: &Layout,
    op: F,
) -> Option<Vec<Out>>
where
    L: Copy,
    R: Copy,
    F: Fn(L, R) -> Out,
{
    let numel = lhs_layout.num_elements();
    if numel == 0 {
        return Some(Vec::new());
    }
    let nest = collapse_for_zip(lhs_layout, rhs_layout)?;

    let mut out: Vec<Out> = Vec::with_capacity(numel);
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
/// The caller must have checked [`ZipNest::lhs_is_dense`] (so
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
pub(crate) fn zip_apply_inplace<D, S, F>(nest: &ZipNest, dst: &mut [D], src: &[S], op: F)
where
    D: Copy,
    S: Copy,
    F: Fn(D, S) -> D,
{
    debug_assert!(
        nest.lhs_is_dense(),
        "zip_apply_inplace: destination must be dense"
    );
    if nest.ndim == 0 {
        // All dims were size 1: a single element.
        dst[nest.lhs_offset] = op(dst[nest.lhs_offset], src[nest.rhs_offset]);
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

// ============================================================================
// 3-way zip support (for mask_where and ternary operations)
// ============================================================================

/// A tuple of three layouts collapsed into a joint loop nest.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Zip3Nest {
    pub ndim: usize,
    pub shape: [usize; ZIP_MAX_RANK],
    pub a_strides: [isize; ZIP_MAX_RANK],
    pub b_strides: [isize; ZIP_MAX_RANK],
    pub c_strides: [isize; ZIP_MAX_RANK],
    pub a_offset: usize,
    pub b_offset: usize,
    pub c_offset: usize,
}

impl Zip3Nest {
    #[inline]
    pub fn inner(&self) -> (usize, isize, isize, isize) {
        let d = self.ndim - 1;
        (
            self.shape[d],
            self.a_strides[d],
            self.b_strides[d],
            self.c_strides[d],
        )
    }

    pub fn a_is_dense(&self) -> bool {
        Self::is_dense(&self.a_strides[..self.ndim], &self.shape[..self.ndim])
    }

    #[allow(dead_code)]
    pub fn b_is_dense(&self) -> bool {
        Self::is_dense(&self.b_strides[..self.ndim], &self.shape[..self.ndim])
    }

    #[allow(dead_code)]
    pub fn c_is_dense(&self) -> bool {
        Self::is_dense(&self.c_strides[..self.ndim], &self.shape[..self.ndim])
    }

    fn is_dense(strides: &[isize], shape: &[usize]) -> bool {
        let mut expected = 1isize;
        for d in (0..shape.len()).rev() {
            if strides[d] != expected {
                return false;
            }
            match expected.checked_mul(shape[d] as isize) {
                Some(next) => expected = next,
                None => return false,
            }
        }
        true
    }

    pub fn for_each_run(&self, mut f: impl FnMut(usize, usize, usize)) {
        debug_assert!(self.ndim >= 1);
        let outer = self.ndim - 1;
        let mut idx = [0usize; ZIP_MAX_RANK];
        let mut a_base = self.a_offset as isize;
        let mut b_base = self.b_offset as isize;
        let mut c_base = self.c_offset as isize;
        loop {
            f(a_base as usize, b_base as usize, c_base as usize);
            let mut d = outer;
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                idx[d] += 1;
                a_base += self.a_strides[d];
                b_base += self.b_strides[d];
                c_base += self.c_strides[d];
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
                a_base -= self.shape[d] as isize * self.a_strides[d];
                b_base -= self.shape[d] as isize * self.b_strides[d];
                c_base -= self.shape[d] as isize * self.c_strides[d];
            }
        }
    }
}

/// Jointly collapse three same-shape layouts into the minimum-rank
/// equivalent loop nest. Supports heterogeneous element types.
pub(crate) fn collapse_for_zip3(a: &Layout, b: &Layout, c: &Layout) -> Option<Zip3Nest> {
    let shape = a.shape();
    let ndims = a.num_dims();
    debug_assert_eq!(
        &shape[..],
        &b.shape()[..],
        "collapse_for_zip3: operands must be broadcast to the same shape"
    );
    debug_assert_eq!(
        &shape[..],
        &c.shape()[..],
        "collapse_for_zip3: operands must be broadcast to the same shape"
    );
    if ndims > ZIP_MAX_RANK {
        return None;
    }
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = c.strides();
    if a_strides
        .iter()
        .chain(b_strides)
        .chain(c_strides)
        .any(|&s| s < 0)
    {
        return None;
    }

    let mut nest = Zip3Nest {
        ndim: 0,
        shape: [0; ZIP_MAX_RANK],
        a_strides: [0; ZIP_MAX_RANK],
        b_strides: [0; ZIP_MAX_RANK],
        c_strides: [0; ZIP_MAX_RANK],
        a_offset: a.start_offset(),
        b_offset: b.start_offset(),
        c_offset: c.start_offset(),
    };

    for d in 0..ndims {
        let size = shape[d];
        if size == 1 {
            continue;
        }
        let a_st = a_strides[d];
        let b_st = b_strides[d];
        let c_st = c_strides[d];
        let merge = nest.ndim > 0 && {
            let prev = nest.ndim - 1;
            (size as isize)
                .checked_mul(a_st)
                .is_some_and(|run| nest.a_strides[prev] == run)
                && (size as isize)
                    .checked_mul(b_st)
                    .is_some_and(|run| nest.b_strides[prev] == run)
                && (size as isize)
                    .checked_mul(c_st)
                    .is_some_and(|run| nest.c_strides[prev] == run)
        };
        if merge {
            nest.shape[nest.ndim - 1] *= size;
            nest.a_strides[nest.ndim - 1] = a_st;
            nest.b_strides[nest.ndim - 1] = b_st;
            nest.c_strides[nest.ndim - 1] = c_st;
        } else {
            nest.shape[nest.ndim] = size;
            nest.a_strides[nest.ndim] = a_st;
            nest.b_strides[nest.ndim] = b_st;
            nest.c_strides[nest.ndim] = c_st;
            nest.ndim += 1;
        }
    }

    Some(nest)
}

/// Apply a 3-way mapping operation over strided layouts into a new Vec.
pub(crate) fn zip3_map<A, B, C, R, F>(
    a: &[A],
    a_layout: &Layout,
    b: &[B],
    b_layout: &Layout,
    c: &[C],
    c_layout: &Layout,
    op: F,
) -> Option<Vec<R>>
where
    A: Copy,
    B: Copy,
    C: Copy,
    F: Fn(A, B, C) -> R,
{
    let numel = a_layout.num_elements();
    if numel == 0 {
        return Some(Vec::new());
    }
    let nest = collapse_for_zip3(a_layout, b_layout, c_layout)?;

    let mut out: Vec<R> = Vec::with_capacity(numel);
    if nest.ndim == 0 {
        out.push(op(a[nest.a_offset], b[nest.b_offset], c[nest.c_offset]));
        return Some(out);
    }

    let (len, a_st, b_st, c_st) = nest.inner();
    match (a_st, b_st, c_st) {
        (1, 1, 1) => nest.for_each_run(|ab, bb, cb| {
            let a_slice = &a[ab..ab + len];
            let b_slice = &b[bb..bb + len];
            let c_slice = &c[cb..cb + len];
            for i in 0..len {
                out.push(op(a_slice[i], b_slice[i], c_slice[i]));
            }
        }),
        _ => nest.for_each_run(|ab, bb, cb| {
            for i in 0..len {
                let av = a[ab + i * a_st as usize];
                let bv = b[bb + i * b_st as usize];
                let cv = c[cb + i * c_st as usize];
                out.push(op(av, bv, cv));
            }
        }),
    }
    debug_assert_eq!(out.len(), numel);
    Some(out)
}

/// Apply a 3-way operation in place into `dst`, reading from `b` and `c`.
/// `dst` is indexed directly at `db`, which includes `a_offset`.
pub(crate) fn zip3_apply_inplace<D, B, C, F>(
    nest: &Zip3Nest,
    dst: &mut [D],
    b: &[B],
    c: &[C],
    op: F,
) where
    D: Copy,
    B: Copy,
    C: Copy,
    F: Fn(D, B, C) -> D,
{
    debug_assert!(
        nest.a_is_dense(),
        "zip3_apply_inplace: destination must be dense"
    );
    if nest.ndim == 0 {
        dst[nest.a_offset] = op(dst[nest.a_offset], b[nest.b_offset], c[nest.c_offset]);
        return;
    }
    if nest.shape[..nest.ndim].contains(&0) {
        return;
    }

    let (len, d_st, b_st, c_st) = nest.inner();
    match (d_st, b_st, c_st) {
        (1, 1, 1) => nest.for_each_run(|db, bb, cb| {
            let b_slice = &b[bb..bb + len];
            let c_slice = &c[cb..cb + len];
            let d_slice = &mut dst[db..db + len];
            for i in 0..len {
                d_slice[i] = op(d_slice[i], b_slice[i], c_slice[i]);
            }
        }),
        _ => nest.for_each_run(|db, bb, cb| {
            for i in 0..len {
                let d_idx = db + i * d_st as usize;
                let b_idx = bb + i * b_st as usize;
                let c_idx = cb + i * c_st as usize;
                dst[d_idx] = op(dst[d_idx], b[b_idx], c[c_idx]);
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
        let got = zip_map::<f32, f32, f32, _>(&[], &l, &[], &r, |a, b| a + b).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn test_zip3_map_broadcasting() {
        // Broadcast shapes [2, 1], [1, 3], and [2, 3] to [2, 3]
        let a_layout = broadcast_layout(&[2, 1], &[2, 3]);
        let b_layout = broadcast_layout(&[1, 3], &[2, 3]);
        let c_layout = Layout::contiguous(Shape::from(vec![2, 3]));

        let a_data = vec![1.0f32, 2.0];
        let b_data = vec![10.0f32, 20.0, 30.0];
        let c_data = vec![100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];

        let got: Vec<f32> = zip3_map(
            &a_data,
            &a_layout,
            &b_data,
            &b_layout,
            &c_data,
            &c_layout,
            |a, b, c| a + b + c,
        )
        .unwrap();

        let expected = vec![
            1.0 + 10.0 + 100.0,
            1.0 + 20.0 + 200.0,
            1.0 + 30.0 + 300.0,
            2.0 + 10.0 + 400.0,
            2.0 + 20.0 + 500.0,
            2.0 + 30.0 + 600.0,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_zip3_apply_inplace_broadcasting() {
        // Inplace destination [2, 3] mutated with broadcast sources [2, 1] and [1, 3]
        let mut dst_data = vec![100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];
        let dst_layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let s1_layout = broadcast_layout(&[2, 1], &[2, 3]);
        let s2_layout = broadcast_layout(&[1, 3], &[2, 3]);

        let s1_data = vec![1.0f32, 2.0];
        let s2_data = vec![10.0f32, 20.0, 30.0];

        let nest = collapse_for_zip3(&dst_layout, &s1_layout, &s2_layout).unwrap();
        assert!(nest.a_is_dense());

        zip3_apply_inplace(&nest, &mut dst_data, &s1_data, &s2_data, |d, s1, s2| {
            d + s1 + s2
        });

        let expected = vec![
            100.0 + 1.0 + 10.0,
            200.0 + 1.0 + 20.0,
            300.0 + 1.0 + 30.0,
            400.0 + 2.0 + 10.0,
            500.0 + 2.0 + 20.0,
            600.0 + 2.0 + 30.0,
        ];
        assert_eq!(dst_data, expected);
    }
}
