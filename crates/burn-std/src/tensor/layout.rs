//! The order a tensor's dimensions occupy memory in.
//!
//! A tensor's strides say which dimension is innermost, which next, and so on
//! outward — its *dimension order*. Contiguous NCHW is the identity order; a
//! convolution that computes channels-last and hands back a permuted view is
//! `[0, 2, 3, 1]`. A kernel that iterates a tensor in its own dimension order
//! reads it linearly; one that iterates in any other order reads it strided.
//!
//! Two questions are asked here. [dim_order] asks for the order of a tensor
//! that is *dense* — that fills its buffer with no gaps — which is what a kernel
//! needs before it may treat the buffer as a flat run of elements.
//! [nested_dim_order] asks only that the dimensions *nest*, tolerating the gap a
//! pitched or tile-aligned allocation leaves under a dimension, which is enough
//! to decide what order to iterate in.

use alloc::vec;
use alloc::vec::Vec;

use crate::Shape;

/// The order a tensor's dimensions appear in memory, outermost first.
///
/// `[0, 1, 2, 3]` is contiguous NCHW; `[0, 2, 3, 1]` is NHWC. This is the same
/// convention as the permutation passed to `Tensor::permute`.
pub type DimOrder = Shape;

/// The dimension order of a tensor that is dense in memory, or `None` if it is
/// not dense.
///
/// Dense means the strides are exactly a permutation of contiguous strides: no
/// gaps, no overlap, no broadcasting. Use [nested_dim_order] when only an
/// iteration order is needed and gaps in storage are acceptable.
///
/// Dimensions of size one are ignored while checking density — their stride is
/// arbitrary and carries no traffic — but they keep a position in the returned
/// order so it stays a permutation of `0..rank`.
pub fn dim_order(shape: &[usize], strides: &[usize]) -> Option<DimOrder> {
    dim_order_inner(shape, strides, Padding::Rejected)
}

/// The dimension order of a tensor whose dimensions nest without overlapping,
/// or `None` if they do not.
///
/// Weaker than [dim_order], which additionally requires consecutive elements
/// without gaps. A dimension may sit at a larger stride than the extents inside it
/// need, which is what a pitched or tile-aligned allocation produces: 48
/// channels held innermost on a 64-element tile have stride 64 where a dense
/// tensor would have 48.
///
/// Iterating in this order can improve locality, but reads must still use the
/// tensor's actual strides. Gaps can affect memory transactions and vectorization;
/// accepting an order does not guarantee dense-access performance. This also
/// accepts sliced views whose dimensions satisfy the same nesting condition.
/// Anything reinterpreting a buffer as a flat run of elements must keep asking
/// [dim_order].
pub fn nested_dim_order(shape: &[usize], strides: &[usize]) -> Option<DimOrder> {
    dim_order_inner(shape, strides, Padding::Allowed)
}

#[derive(Clone, Copy)]
enum Padding {
    Allowed,
    Rejected,
}

fn dim_order_inner(shape: &[usize], strides: &[usize], padding: Padding) -> Option<DimOrder> {
    let rank = shape.len();

    if rank != strides.len() {
        return None;
    }

    let mut order: Vec<usize> = (0..rank).collect();
    // Descending stride is outermost first. The dimension index breaks ties so
    // that equal strides — which only happens among size-one dimensions — give
    // a deterministic order rather than one that depends on the sort.
    order.sort_by(|a, b| strides[*b].cmp(&strides[*a]).then(a.cmp(b)));

    let mut expected = 1;

    for &axis in order.iter().rev() {
        if shape[axis] == 1 {
            continue;
        }
        match padding {
            // A gap is what makes the tensor padded rather than dense; an
            // overlap is not a layout at all.
            Padding::Allowed if strides[axis] < expected => return None,
            Padding::Rejected if strides[axis] != expected => return None,
            _ => {}
        }
        // Where the stride is exactly `expected` this is `expected *= shape[axis]`,
        // so the dense walk is the padded one with the gaps taken out.
        expected = strides[axis] * shape[axis];
    }

    Some(Shape::from(order))
}

/// Whether a dimension order is the contiguous one, `[0, 1, .., rank - 1]`.
pub fn is_contiguous_order(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(pos, axis)| pos == *axis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_is_the_identity_order() {
        let shape = [2, 48, 16, 16];
        let strides = [48 * 16 * 16, 16 * 16, 16, 1];

        assert_eq!(
            dim_order(&shape, &strides),
            Some(Shape::from(vec![0, 1, 2, 3]))
        );
    }

    #[test]
    fn nhwc_memory_gives_the_nhwc_order() {
        // What a convolution hands its successor: shape is NCHW, memory is NHWC.
        let shape = [2, 48, 16, 16];
        let strides = [16 * 16 * 48, 1, 16 * 48, 48];

        assert_eq!(
            dim_order(&shape, &strides),
            Some(Shape::from(vec![0, 2, 3, 1]))
        );
    }

    #[test]
    fn broadcast_is_not_dense() {
        let shape = [2, 48, 16, 16];
        let strides = [0, 1, 0, 0];

        assert_eq!(dim_order(&shape, &strides), None);
    }

    #[test]
    fn a_slice_is_not_dense() {
        // A view into a wider tensor: the row stride overshoots the row.
        let shape = [4, 8];
        let strides = [16, 1];

        assert_eq!(dim_order(&shape, &strides), None);
    }

    #[test]
    fn size_one_dimensions_do_not_decide_density() {
        // A per-channel parameter presented at full rank. The strides of the
        // degenerate dimensions say nothing, and must not make it non-dense.
        let shape = [1, 48, 1, 1];
        let strides = [48, 1, 48, 48];

        assert!(dim_order(&shape, &strides).is_some());
    }

    #[test]
    fn the_order_ends_at_the_innermost_dimension() {
        let shape = [2, 48, 16, 16];

        let contiguous = dim_order(&shape, &[48 * 16 * 16, 16 * 16, 16, 1]).unwrap();
        let nhwc = dim_order(&shape, &[16 * 16 * 48, 1, 16 * 48, 48]).unwrap();

        assert_eq!(contiguous.last(), Some(&3));
        assert_eq!(nhwc.last(), Some(&1));
    }

    #[test]
    fn order_is_a_permutation() {
        assert!(is_contiguous_order(&[0, 1, 2, 3]));
        assert!(!is_contiguous_order(&[0, 2, 3, 1]));
    }

    #[test]
    fn a_dense_tensor_nests_in_the_order_it_is_dense_in() {
        let shape = [2, 48, 16, 16];

        for strides in [
            [48 * 16 * 16, 16 * 16, 16, 1],
            [16 * 16 * 48, 1, 16 * 48, 48],
        ] {
            let dense = dim_order(&shape, &strides);
            assert!(dense.is_some());
            assert_eq!(nested_dim_order(&shape, &strides), dense);
        }
    }

    #[test]
    fn padding_under_the_innermost_dimension_keeps_the_nhwc_order() {
        let shape = [2, 48, 16, 16];
        let strides = [16 * 16 * 64, 1, 16 * 64, 64];

        assert_eq!(dim_order(&shape, &strides), None);
        assert_eq!(
            nested_dim_order(&shape, &strides),
            Some(Shape::from(vec![0, 2, 3, 1]))
        );
    }

    #[test]
    fn padding_above_the_innermost_dimension_is_nesting_too() {
        let shape = [4, 8];
        let strides = [16, 1];

        assert_eq!(dim_order(&shape, &strides), None);
        assert_eq!(
            nested_dim_order(&shape, &strides),
            Some(Shape::from(vec![0, 1]))
        );
    }

    #[test]
    fn overlapping_dimensions_are_not_an_order_under_either() {
        let shape = [4, 8];
        let strides = [4, 1];

        assert_eq!(dim_order(&shape, &strides), None);
        assert_eq!(nested_dim_order(&shape, &strides), None);
    }

    #[test]
    fn a_broadcast_dimension_still_cannot_vote() {
        let shape = [2, 48, 16, 16];
        let strides = [0, 1, 0, 0];

        assert_eq!(nested_dim_order(&shape, &strides), None);
    }

    #[test]
    fn size_one_dimensions_neither_pad_nor_constrain_nesting() {
        let shape = [1, 48, 1, 1];
        let strides = [48, 1, 48, 48];

        assert_eq!(
            nested_dim_order(&shape, &strides),
            dim_order(&shape, &strides)
        );
        assert!(nested_dim_order(&shape, &strides).is_some());
    }
}
