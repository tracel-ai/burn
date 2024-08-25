use crate::{backend::Backend, check::TensorCheck, Dense, Float, Sparse, Tensor, TensorKind};
use crate::{check, Bool, SparseStorage, TensorPrimitive, TensorRepr};

impl<const D: usize, B, SR> Tensor<B, D, Float, Sparse<B, SR>>
where
    B: Backend,
    SR: SparseStorage<B>,
    (B, Float, Sparse<B, SR>): TensorRepr,
{
    /// Executes an operation on the tensor and modifies its value.
    ///
    /// # Notes
    ///
    /// This won't necessary reuse the same tensor data/buffer, but it should if there is
    /// no other reference pointing to the same tensor.
    ///
    /// Wrapping operations with inplace is not an optimization, it's mainly there if you
    /// want to mutate a tensor by using owned operations. A plausible usage would be to
    /// update the weights of a mutable model reference.
    pub fn inplace<F: FnOnce(Self) -> Self>(&mut self, func: F) {
        let mut tensor_owned = Tensor::empty([0; D], &self.device());
        core::mem::swap(&mut tensor_owned, self);

        let mut tensor_new = func(tensor_owned);
        core::mem::swap(&mut tensor_new, self);
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors dont' have a compatible shape.
    pub fn spmm(self, rhs: Tensor<B, D, Float, Dense>) -> Tensor<B, D, Float, Dense> {
        // check!(TensorCheck::spmm(&self, &rhs));
        Tensor::<B, D, Float, Dense>::new(TensorPrimitive::Float(SR::float_spmm(
            self.into_primitive(),
            rhs.into_primitive(),
        )))
    }

    pub fn sddmm(self, lhs: Tensor<B, D, Float, Dense>, rhs: Tensor<B, D, Float, Dense>) -> Self {
        Tensor::new(SR::float_sddmm(
            lhs.into_primitive().tensor(),
            rhs.into_primitive().tensor(),
            self.into_primitive(),
        ))
    }
}
