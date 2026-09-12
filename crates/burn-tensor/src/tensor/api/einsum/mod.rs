//! Einstein summation through existing differentiable tensor operations.

use core::marker::PhantomData;

use crate::{Float, Tensor, kind::Numeric, ops::BridgeTensor};

mod execution;

/// An einsum operand whose rank is determined at runtime.
///
/// Convert owned tensors with `.into()`, or borrow them with `(&tensor).into()`.
/// All operands must have the same tensor kind, dtype and device.
#[derive(Clone, Debug)]
pub struct EinsumOperand<K: Numeric = Float> {
    primitive: BridgeTensor,
    kind: PhantomData<K>,
}

impl<const D: usize, K: Numeric> From<Tensor<D, K>> for EinsumOperand<K> {
    fn from(tensor: Tensor<D, K>) -> Self {
        Self {
            primitive: tensor.primitive,
            kind: PhantomData,
        }
    }
}

impl<const D: usize, K: Numeric> From<&Tensor<D, K>> for EinsumOperand<K> {
    fn from(tensor: &Tensor<D, K>) -> Self {
        tensor.clone().into()
    }
}

impl<const D: usize, K: Numeric> Tensor<D, K> {
    /// Computes Einstein summation with a runtime equation.
    ///
    /// Supports explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`) outputs,
    /// any number of operands, repeated-label diagonals, and broadcasting.
    /// Labels are case-sensitive ASCII letters. Implicit output puts ellipsis
    /// dimensions first, followed by labels occurring exactly once, in `A-Z`,
    /// `a-z` order. Ellipsis dimensions broadcast from the right and may be
    /// reduced by omitting them from an explicit output.
    ///
    /// Operands are contracted from left to right using multiplication,
    /// reductions, and batched matrix multiplication. No optimized contraction
    /// path is searched. Float operations retain automatic differentiation.
    ///
    /// Burn represents scalars as rank-one tensors with shape `[1]`: a scalar
    /// result requires `D = 1`, and an empty input subscript accepts shape `[1]`.
    /// Other inputs must have exactly the rank described by their subscripts.
    ///
    /// For a literal equation, [`einsum!`](crate::einsum) also validates the
    /// equation and statically determined ranks, and generates the operation chain
    /// from the shared contraction plan at compile time.
    ///
    /// # Example
    ///
    /// ```
    /// use burn_tensor::Tensor;
    /// let device = Default::default();
    /// let matrix = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
    /// let vector = Tensor::<1>::from_floats([10., 20.], &device);
    /// let result = Tensor::<1>::einsum("ij,j->i", [matrix.into(), vector.into()]);
    /// assert_eq!(result.into_data().try_to_vec::<f32>().unwrap(), [50., 110.]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for an invalid equation, no operands, incompatible ranks or
    /// dimensions, unequal repeated-label dimensions, mismatched devices or
    /// dtypes, quantized operands, or an incorrect output rank `D`.
    pub fn einsum(equation: &str, operands: impl IntoIterator<Item = EinsumOperand<K>>) -> Self {
        let equation =
            burn_einsum::parse(equation).unwrap_or_else(|error| panic!("einsum: {error}"));
        execution::execute(&equation.plan(), operands)
    }
}

/// Implementation details used by the generated `einsum!` operation chain.
#[doc(hidden)]
pub mod __einsum {
    pub use super::execution::{
        alignment_shape, axes, broadcast_contract, can_contract_early, can_matmul, contract_early,
        matmul_shapes, prepare, validate_broadcast,
    };
    pub use burn_einsum::Axis;
}
