use crate::{backend::Backend, check::TensorCheck, Dense, Float, Sparse, Tensor, TensorKind};
use crate::{check, BasicOps, BasicSparseOps, Bool, SparseStorage, TensorPrimitive, TensorRepr};

impl<const D: usize, B, K> Tensor<B, D, K, Dense>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub fn into_sparse<SR: SparseStorage<B> + BasicSparseOps<B, K, SR>>(
        self,
    ) -> Tensor<B, D, K, Sparse<B, SR>>
    where
        K: BasicOps<B, Sparse<B, SR>>,
        (B, K, Sparse<B, SR>): TensorRepr,
    {
        Tensor::<B, D, K, Sparse<B, SR>>::from_primitive(SR::into_sparse(self.primitive))
    }
}

impl<const D: usize, B, K, SR> Tensor<B, D, K, Sparse<B, SR>>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B, Sparse<B, SR>> + BasicOps<B, Dense>,
    SR: SparseStorage<B> + BasicSparseOps<B, K, SR>,
    (B, K, Sparse<B, SR>): TensorRepr,
{
    pub fn into_dense(self) -> Tensor<B, D, K, Dense> {
        Tensor::<B, D, K, Dense>::from_primitive(SR::into_dense(self.primitive))
    }
}
