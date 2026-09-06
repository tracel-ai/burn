//! Einstein summation through existing differentiable tensor operations.

use alloc::{collections::VecDeque, vec::Vec};
use core::marker::PhantomData;

use burn_einsum::Equation;

use crate::{Float, Tensor, kind::Numeric, ops::BridgeTensor};

mod execution;
use execution::{Layout, align, contract_pair, finalize};

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
    /// equation and statically determined ranks at compile time.
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
        let mut execution = Execution::from_equation(equation, operands);
        execution.check_output_rank::<D>();
        while execution.operands.len() > 1 {
            execution = execution.contract_next();
        }
        execution.finish()
    }
}

/// Implementation details used by the generated `einsum!` operation chain.
#[doc(hidden)]
pub mod __einsum {
    pub use super::Execution;
}

/// A partially executed contraction, used by the literal macro expansion.
#[doc(hidden)]
pub struct Execution<K: Numeric> {
    operands: VecDeque<BridgeTensor>,
    layout: Layout,
    dimension_counts: Vec<usize>,
    kind: PhantomData<K>,
}

impl<K: Numeric> Execution<K> {
    /// Starts a contraction from the macro's already parsed label arrays.
    pub fn new(
        inputs: &[&[u8]],
        output: &[u8],
        operands: impl IntoIterator<Item = EinsumOperand<K>>,
    ) -> Self {
        Self::from_equation(
            Equation {
                inputs: inputs.iter().map(|labels| labels.to_vec()).collect(),
                output: output.to_vec(),
            },
            operands,
        )
    }

    fn from_equation(
        equation: Equation,
        operands: impl IntoIterator<Item = EinsumOperand<K>>,
    ) -> Self {
        let operands: Vec<_> = operands.into_iter().map(|op| op.primitive).collect();
        let layout = Layout::new::<K>(&equation, &operands);
        let (operands, dimension_counts) = align::<K>(&equation, &layout, operands);
        Self {
            operands,
            layout,
            dimension_counts,
            kind: PhantomData,
        }
    }

    /// Contracts the next pair while retaining axes needed by later operands.
    pub fn contract_next(mut self) -> Self {
        assert!(
            self.operands.len() >= 2,
            "einsum: no remaining operand pair"
        );
        let left = self.operands.pop_front().unwrap();
        let right = self.operands.pop_front().unwrap();
        let result = contract_pair::<K>(left, right, &self.layout, &mut self.dimension_counts);
        self.operands.push_front(result);
        self
    }

    fn check_output_rank<const D: usize>(&self) {
        assert_eq!(
            D,
            self.layout.output_dimensions.max(1),
            "einsum: output rank does not match the equation (scalar results have rank 1)"
        );
    }

    /// Reduces remaining unary axes and restores the typed output.
    pub fn finish<const D: usize>(mut self) -> Tensor<D, K> {
        self.check_output_rank::<D>();
        assert_eq!(self.operands.len(), 1, "einsum: unfinished contraction");
        Tensor::new(finalize::<K>(
            self.operands.pop_front().unwrap(),
            &self.layout,
        ))
    }
}
