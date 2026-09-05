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
//! then allocated dense in that order, an operand already in it is walked
//! linearly and vectorized, one that disagrees is walked strided exactly as
//! before, and the result is permuted back to the logical order before it is
//! handed on. A permute is a change of metadata, so none of this moves data.

use burn_std::{
    Shape,
    tensor::layout::{DimOrder, is_contiguous_order, nested_dim_order},
};

use crate::{ops::permute, tensor::CubeTensor};

/// Runs `launch` with the operands permuted into their memory order, and
/// permutes what it returns back.
///
/// `output_shape` is the shape the kernel will write — the broadcast of the
/// operands for a binary operation, the operand's own for a unary one. Only
/// operands of exactly that shape vote on the order: a broadcast operand is
/// read from cache whatever the order is, so it has no stake, and it permutes
/// along without changing whether it broadcasts.
pub(crate) fn in_memory_order<const N: usize>(
    operands: [CubeTensor; N],
    output_shape: &Shape,
    launch: impl FnOnce([CubeTensor; N]) -> CubeTensor,
) -> CubeTensor {
    let Some(order) = MemoryOrder::of(&operands, output_shape) else {
        return launch(operands);
    };

    let presented = operands.map(|operand| order.present(operand));

    order.restore(launch(presented))
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

        let mut votes: Vec<(DimOrder, usize)> = Vec::new();

        for operand in operands {
            if operand.meta.shape() != output_shape {
                continue;
            }
            let Some(order) = nested_dim_order(operand.meta.shape(), operand.meta.strides()) else {
                continue;
            };
            let bytes = operand.meta.num_elements() * operand.dtype.size();

            match votes.iter_mut().find(|(candidate, _)| candidate == &order) {
                Some((_, total)) => *total += bytes,
                None => votes.push((order, bytes)),
            }
        }

        let (winner, _) = votes
            .into_iter()
            .max_by_key(|(order, bytes)| (*bytes, is_contiguous_order(order)))?;

        if is_contiguous_order(&winner) {
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
