//! Walking an elementwise kernel in the order its operands occupy memory.
//!
//! The elementwise kernels take every operand through a linear view that walks
//! the *logical* dimension order, and allocate their output contiguous in that
//! order. That is the right order only when the operands are laid out that way.
//! A convolution hands its consumer a channels-last tensor presented in NCHW,
//! and such a kernel then walks it strided — and because the logical last
//! dimension is not the contiguous one, it cannot vectorize either.
//!
//! An elementwise operation does not care what a dimension means, only that
//! every operand is walked in step. So the operands are handed to the kernel
//! permuted into the order the bulk of them occupy memory in. The output is
//! allocated dense in that order when a fresh buffer is needed; in-place writes
//! retain the operand's storage. Matching dense operands can be read linearly,
//! while vectorization still depends on the strides and all operands. The result
//! is permuted back to logical order. Permutation changes metadata, not data.
//! This is a locality heuristic, not a guarantee of faster execution for every
//! shape or consumer. A fresh result can retain a nonlogical physical layout;
//! a later reshape that cannot be expressed through strides must materialize
//! logical value order. The extra copy can outweigh the elementwise speedup.
//! Choosing the order here cannot account for consumers that have not run yet.
//!
//! A storage-tiled operand is walked the same way, one step further: its buffer is a strided
//! tensor of more dims, the fragments its tiling splits each dim into. The kernel walks those
//! physical dims, every other operand restated over them by strides, and the output takes the
//! tiling back. A tiled tensor goes through an elementwise operation with no relayout and comes
//! out tiled; an optimizer step keeps a tiled weight tiled.

use burn_backend::TensorMetadata;
use burn_std::{
    Metadata, Shape,
    tensor::layout::{DimOrder, is_contiguous_order, nested_dim_order},
};

use crate::{
    kernel::untile,
    ops::{expand, permute},
    tensor::CubeTensor,
};

/// Runs `launch` with the operands permuted into their memory order, and
/// permutes what it returns back.
///
/// `output_shape` is the shape the kernel will write — the broadcast of the
/// operands for a binary operation, the operand's own for a unary one. Only
/// operands of exactly that shape vote on the order. This favors full-sized
/// inputs over broadcast parameters; large broadcast inputs may still incur
/// significant traffic. Every operand permutes along with the output shape.
///
/// A storage-tiled operand of that shape first sets the dims walked: its physical ones
/// ([`StorageTiling`]).
pub(crate) fn in_memory_order<const N: usize>(
    operands: [CubeTensor; N],
    output_shape: Shape,
    launch: impl FnOnce([CubeTensor; N], Shape) -> CubeTensor,
) -> CubeTensor {
    if let Some(tiling) = StorageTiling::new(&operands, &output_shape) {
        let presented = operands.map(|operand| tiling.present(operand));
        let physical = tiling.stored.shape.clone();
        return tiling.restore(in_memory_order(presented, physical, launch));
    }

    let Some(order) = MemoryOrder::of(&operands, &output_shape) else {
        return launch(operands, output_shape);
    };

    let presented = operands.map(|operand| order.present(operand));

    let shape = Shape::from(
        order
            .axes
            .iter()
            .map(|&axis| output_shape[axis])
            .collect::<Vec<_>>(),
    );
    order.restore(launch(presented, shape))
}

/// The storage tiles an elementwise launch walks: those of the first storage-tiled operand of the
/// output's shape. Each operand is presented as a plain tensor of its physical dims: one stored
/// the same way as it lies, any other in rows and restated over those dims by strides
/// (`Metadata::tiled_like`), so every operand walks the same values in step.
struct StorageTiling {
    /// How the tiled operand is stored.
    stored: Metadata,
    /// The output's logical shape, which every operand is broadcast to.
    logical: Shape,
}

impl StorageTiling {
    /// The tiling the operands are walked in, or `None` where none of the output's shape is
    /// tiled. A quantized operand keeps its own rules, as [`MemoryOrder`] leaves it.
    fn new(operands: &[CubeTensor], output_shape: &Shape) -> Option<Self> {
        if operands.iter().any(|operand| operand.qparams.is_some()) {
            return None;
        }
        let tiled = operands
            .iter()
            .find(|operand| operand.meta.is_tiled() && &operand.shape() == output_shape)?;
        Some(Self {
            stored: (*tiled.meta).clone(),
            logical: output_shape.clone(),
        })
    }

    /// `operand` as a plain tensor over the physical dims of the tiles: its own buffer where it
    /// is stored in them, otherwise its rows broadcast to the output and split by strides. An
    /// operand tiled another way is laid back in rows first.
    fn present(&self, operand: CubeTensor) -> CubeTensor {
        let stored_alike =
            operand.meta.tiling == self.stored.tiling && operand.meta.shape == self.stored.shape;
        let operand = match stored_alike {
            true => operand,
            false => {
                let rows = expand(untile(operand), self.logical.clone());
                let meta = rows
                    .meta
                    .tiled_like(&self.stored)
                    .expect("an operand broadcast to the output's shape splits into its tiles");
                CubeTensor {
                    meta: Box::new(meta),
                    ..rows
                }
            }
        };
        let meta = Metadata::new(operand.meta.shape.clone(), operand.meta.strides.clone());
        CubeTensor {
            meta: Box::new(meta),
            ..operand
        }
    }

    /// The kernel's output, written over the physical dims, labelled with the tiling again.
    fn restore(&self, output: CubeTensor) -> CubeTensor {
        let meta = (*output.meta)
            .clone()
            .with_tiling(self.stored.tiling)
            .expect("the output is written over the tiles' physical dims");
        CubeTensor {
            meta: Box::new(meta),
            ..output
        }
    }
}

/// The permutation that presents a set of operands in their memory order, and
/// its inverse.
struct MemoryOrder {
    axes: Vec<usize>,
    inverse: Vec<usize>,
}

impl MemoryOrder {
    /// The order the operands carrying the most bytes occupy memory in, or
    /// `None` when that is the logical order already — the case that should
    /// keep behaving exactly as it always has.
    ///
    /// Ties go to the logical order, so a pair of operands that disagree keeps
    /// the contiguous walk rather than favouring whichever came first. Padding
    /// under a dimension is tolerated: a pitched allocation leaves one, and
    /// what is being decided is the walk order, which padding does not touch.
    fn of(operands: &[CubeTensor], output_shape: &Shape) -> Option<Self> {
        // A quantized operand's block layout has its own rules for a permute;
        // leaving it in logical order costs nothing it was not already paying.
        if operands.iter().any(|operand| operand.qparams.is_some()) {
            return None;
        }

        Self::from_layouts(
            operands.iter().map(|operand| {
                (
                    operand.meta.shape().as_slice(),
                    &operand.meta.strides()[..],
                    operand.dtype.size(),
                )
            }),
            output_shape,
        )
    }

    fn from_layouts<'a>(
        layouts: impl Iterator<Item = (&'a [usize], &'a [usize], usize)> + Clone,
        output_shape: &[usize],
    ) -> Option<Self> {
        // The common logical walk needs neither sorting nor allocation. Ignore
        // singleton strides: expanding [1024] to [1, 1024] gives [0, 1], which
        // already has the right walk and must keep its vectorizable last axis.
        if layouts
            .clone()
            .all(|(shape, strides, _)| shape != output_shape || nests_logically(shape, strides))
        {
            return None;
        }

        let mut votes: Vec<(DimOrder, usize)> = Vec::new();
        for (shape, strides, element_size) in layouts {
            if shape != output_shape {
                continue;
            }
            let Some(mut order) = nested_dim_order(shape, strides) else {
                continue;
            };
            if nests_logically(shape, strides) {
                order = Shape::from((0..shape.len()).collect::<Vec<_>>());
            } else {
                // Singleton placement carries no traffic. Put these axes first
                // in logical order, so equivalent walks share a vote and a real
                // innermost axis remains available for vectorization.
                order.sort_by_key(|&axis| {
                    (shape[axis] != 1, if shape[axis] == 1 { axis } else { 0 })
                });
            }
            let bytes = shape.iter().product::<usize>() * element_size;
            match votes.iter_mut().find(|(candidate, _)| candidate == &order) {
                Some((_, total)) => *total += bytes,
                None => votes.push((order, bytes)),
            }
        }

        let max_bytes = votes.iter().map(|(_, bytes)| *bytes).max()?;
        let mut winners = votes.into_iter().filter(|(_, bytes)| *bytes == max_bytes);
        let (winner, _) = winners.next()?;
        if winners.next().is_some() || is_contiguous_order(&winner) {
            return None;
        }

        let axes: Vec<usize> = winner.iter().copied().collect();
        let mut inverse = vec![0; axes.len()];
        for (position, &axis) in axes.iter().enumerate() {
            inverse[axis] = position;
        }

        Some(Self { axes, inverse })
    }

    fn present(&self, tensor: CubeTensor) -> CubeTensor {
        permute(tensor, &self.axes)
    }

    fn restore(&self, tensor: CubeTensor) -> CubeTensor {
        permute(tensor, &self.inverse)
    }
}

/// Whether non-singleton dimensions nest in logical order, with padding allowed.
fn nests_logically(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1;
    for (&size, &stride) in shape.iter().zip(strides).rev() {
        if size == 1 {
            continue;
        }
        if stride < expected {
            return false;
        }
        expected = stride * size;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_strides_do_not_reorder_a_logical_walk() {
        for strides in [[0, 1], [1, 1], [1024, 1]] {
            assert!(
                MemoryOrder::from_layouts(
                    [([1, 1024].as_slice(), strides.as_slice(), 4)].into_iter(),
                    &[1, 1024],
                )
                .is_none()
            );
        }
    }

    #[test]
    fn singleton_axes_cannot_displace_the_innermost_data_axis() {
        let shape = [4, 1, 8];
        for strides in [[1, 0, 4], [1, 1, 4], [1, 32, 4]] {
            let order = MemoryOrder::from_layouts(
                [(shape.as_slice(), strides.as_slice(), 4)].into_iter(),
                &shape,
            )
            .unwrap();
            assert_eq!(order.axes, [1, 2, 0]);
            let presented_shape = Shape::from(
                order
                    .axes
                    .iter()
                    .map(|&axis| shape[axis])
                    .collect::<Vec<_>>(),
            );
            let presented_strides = burn_std::Strides::from(
                order
                    .axes
                    .iter()
                    .map(|&axis| strides[axis])
                    .collect::<Vec<_>>(),
            );
            // A zero singleton stride must not turn the last axis into a
            // broadcast axis. Other arbitrary strides can still limit alignment.
            if strides[1] == 0 {
                assert_eq!(
                    cubecl::tensor_vector_size_parallel(
                        [4, 2, 1].into_iter(),
                        &presented_shape,
                        &presented_strides,
                        2,
                    ),
                    4
                );
            }
            for axis in 0..shape.len() {
                assert_eq!(order.axes[order.inverse[axis]], axis);
            }
        }
    }

    #[test]
    fn conflicting_nonlogical_votes_tie_in_either_operand_order() {
        let shape = [2, 3, 4];
        let layouts = [
            (shape.as_slice(), [12, 1, 3].as_slice(), 4),
            (shape.as_slice(), [4, 8, 1].as_slice(), 4),
        ];
        assert!(MemoryOrder::from_layouts(layouts.into_iter(), &shape).is_none());
        assert!(MemoryOrder::from_layouts(layouts.into_iter().rev(), &shape).is_none());
    }

    #[test]
    fn a_logical_and_nonlogical_vote_tie() {
        let shape = [2, 3, 4];
        assert!(
            MemoryOrder::from_layouts(
                [
                    (shape.as_slice(), [12, 1, 3].as_slice(), 4),
                    (shape.as_slice(), [12, 4, 1].as_slice(), 4),
                ]
                .into_iter(),
                &shape
            )
            .is_none()
        );
    }

    #[test]
    fn equivalent_singleton_layouts_share_their_vote() {
        let shape = [4, 1, 8];
        let order = MemoryOrder::from_layouts(
            [
                (shape.as_slice(), [1, 0, 4].as_slice(), 4),
                (shape.as_slice(), [1, 32, 4].as_slice(), 4),
                (shape.as_slice(), [8, 8, 1].as_slice(), 4),
            ]
            .into_iter(),
            &shape,
        )
        .unwrap();
        assert_eq!(order.axes, [1, 2, 0]);
    }

    #[test]
    fn votes_are_weighted_by_bytes() {
        let shape = [2, 3, 4];
        let order = MemoryOrder::from_layouts(
            [
                (shape.as_slice(), [12, 1, 3].as_slice(), 4),
                (shape.as_slice(), [12, 4, 1].as_slice(), 2),
            ]
            .into_iter(),
            &shape,
        )
        .unwrap();
        assert_eq!(order.axes, [0, 2, 1]);
    }

    #[test]
    fn padded_layouts_vote_but_broadcast_and_overlapping_layouts_do_not() {
        let shape = [2, 3, 4];
        let order = MemoryOrder::from_layouts(
            [
                (shape.as_slice(), [32, 1, 8].as_slice(), 4),
                (shape.as_slice(), [0, 4, 1].as_slice(), 4),
                (shape.as_slice(), [1, 1, 1].as_slice(), 4),
                ([1, 3, 4].as_slice(), [12, 4, 1].as_slice(), 8),
            ]
            .into_iter(),
            &shape,
        )
        .unwrap();
        assert_eq!(order.axes, [0, 2, 1]);
    }
}
