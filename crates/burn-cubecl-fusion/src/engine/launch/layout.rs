//! Choosing which layout a fused block iterates in.
//!
//! A block's reference layout decides how a global index maps to coordinates,
//! and therefore which tensors are read linearly and which are read strided.
//! Outputs used to be allocated contiguous in logical dimension order and the
//! reference taken from them, which is the right answer only when the inputs
//! are contiguous too.
//!
//! They often are not. Every convolution in `burn-cubecl` permutes to NHWC,
//! convolves, and permutes back, and a permute is a metadata change — so a
//! convolution hands its successor an NCHW *view* of NHWC memory. Iterating
//! that in NCHW order reads it strided, and on a tensor too large for the last
//! level cache that costs an order of magnitude in bandwidth.
//!
//! Nothing forces the output to be contiguous. Writing it in any dense layout
//! costs the same, as long as the reference iterates in that layout's memory
//! order. So the layout is free to choose, and the cost of a choice is the
//! traffic of the inputs that disagree with it.

use burn_std::{Shape, Strides};

/// The order a tensor's dimensions appear in memory, outermost first.
///
/// `[0, 1, 2, 3]` is contiguous NCHW; `[0, 2, 3, 1]` is NHWC. This is the same
/// convention as the permutation passed to `Tensor::permute`.
pub type DimOrder = Shape;

/// The dimension order of a tensor that is dense in memory, or `None` if it is
/// not dense.
///
/// Dense means the strides are exactly a permutation of contiguous strides: no
/// gaps, no overlap, no broadcasting. A tensor that is not dense cannot be
/// described by a dimension order, and a block cannot adopt its layout.
///
/// Dimensions of size one are ignored while checking density — their stride is
/// arbitrary and carries no traffic — but they keep a position in the returned
/// order so it stays a permutation of `0..rank`.
pub fn dim_order(shape: &[usize], strides: &[usize]) -> Option<DimOrder> {
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

    for &dim in order.iter().rev() {
        if shape[dim] == 1 {
            continue;
        }
        if strides[dim] != expected {
            return None;
        }
        expected *= shape[dim];
    }

    Some(Shape::from(order))
}

/// The strides a tensor of this shape has when laid out in the given dimension
/// order.
///
/// The inverse of [dim_order] for dense tensors: `strides_for(shape,
/// dim_order(shape, strides))` reproduces `strides` up to the arbitrary strides
/// of size-one dimensions.
pub fn strides_for(shape: &[usize], order: &[usize]) -> Strides {
    let mut strides = vec![0usize; shape.len()];
    let mut current = 1;

    for &dim in order.iter().rev() {
        strides[dim] = current;
        current *= shape[dim];
    }

    Strides::new(&strides)
}

/// Whether a dimension order is the contiguous one, `[0, 1, .., rank - 1]`.
pub fn is_contiguous_order(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(pos, dim)| pos == *dim)
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
    fn round_trips_through_strides() {
        let shape = [2, 48, 16, 16];
        let order = [0, 2, 3, 1];
        let strides = strides_for(&shape, &order);

        assert_eq!(&*strides, &[16 * 16 * 48, 1, 16 * 48, 48]);
        assert_eq!(
            dim_order(&shape, &strides),
            Some(Shape::from(order.to_vec()))
        );
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
}
