use crate::backend::{Sparse, SparseBackend};
use burn_tensor::{Int, Tensor, TensorPrimitive};

pub trait ToSparse<const D: usize, B>
where
    B: SparseBackend,
{
    fn into_sparse(self) -> Tensor<B, D, Sparse>;
}

pub trait SparseTensorApi<const D: usize, B>
where
    B: SparseBackend,
{
    fn dense_int(self) -> Tensor<B, D, Int>;
    fn spmm(self, rhs: Tensor<B, D>) -> Tensor<B, D>;
    fn dense(self) -> Tensor<B, D>;
}

impl<const D: usize, B> ToSparse<D, B> for Tensor<B, D>
where
    B: SparseBackend,
{
    fn into_sparse(self) -> Tensor<B, D, Sparse> {
        Tensor::new(B::sparse_to_sparse(self.into_primitive().tensor()))
    }
}

impl<const D: usize, B> SparseTensorApi<D, B> for Tensor<B, D, Sparse>
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
