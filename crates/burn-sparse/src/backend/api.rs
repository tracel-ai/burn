use crate::backend::{Sparse, SparseBackend};
use burn_tensor::{Int, Tensor, TensorPrimitive};

pub trait SparseTensor<const D: usize, B>
where
    B: SparseBackend,
{
    fn dense_int(self) -> Tensor<B, D, Int>;
    fn spmm(self, rhs: Tensor<B, D>) -> Tensor<B, D>;
    fn dense(self) -> Tensor<B, D>;
}

impl<const D: usize, B> SparseTensor<D, B> for Tensor<B, D, Sparse>
where
    B: SparseBackend,
{
    fn dense(self) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(B::sparse_to_dense(
            self.into_primitive(),
        )))
    }

    fn dense_int(self) -> Tensor<B, D, Int> {
        self.dense().int()
    }

    fn spmm(self, rhs: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(B::sparse_spmm(
            self.into_primitive(),
            rhs.into_primitive().tensor(),
        )))
    }
}
