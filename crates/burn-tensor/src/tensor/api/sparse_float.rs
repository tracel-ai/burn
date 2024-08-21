use crate::check;
use crate::{
    backend::Backend, check::TensorCheck, Dense, Float, Sparse, SparseRepr, Tensor, TensorKind,
};

impl<const D: usize, B, R> Tensor<B, D, Float, Sparse<R, B>>
where
    B: Backend,
    R: SparseRepr<B>,
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
    pub fn spmm(self, rhs: Tensor<B, D, Float, Dense>) -> Self {
        check!(TensorCheck::spmm(&self, &rhs));
        Self::new(R::float_spmm(self.primitive, rhs.primitive))
    }
}
