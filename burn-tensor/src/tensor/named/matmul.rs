use crate::backend::Backend;
use crate::{Dim, NamedDims, NamedTensor, Tensor};

pub trait Matmul<Rhs, Out> {
    fn matmul(self, rhs: Rhs) -> Out;
}

impl<B: Backend, const D: usize, ND> NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors dont' have a compatible shape.
    pub fn matmul<NamedDimsRhs, NamedDimsOut>(
        self,
        rhs: NamedTensor<B, NamedDimsRhs>,
    ) -> NamedTensor<B, NamedDimsOut>
    where
        NamedDimsRhs: NamedDims<B, Tensor = Tensor<B, D>>,
        NamedDimsOut: NamedDims<B, Tensor = Tensor<B, D>>,
        Self: Matmul<NamedTensor<B, NamedDimsRhs>, NamedTensor<B, NamedDimsOut>>,
    {
        Matmul::matmul(self, rhs)
    }
}

impl<B: Backend, X: Dim, Y: Dim, Z: Dim> Matmul<NamedTensor<B, (Y, Z)>, NamedTensor<B, (X, Z)>>
    for NamedTensor<B, (X, Y)>
{
    fn matmul(self, rhs: NamedTensor<B, (Y, Z)>) -> NamedTensor<B, (X, Z)> {
        NamedTensor::from_tensor(self.tensor.matmul(rhs.tensor))
    }
}

impl<B: Backend, Batch: Dim, X: Dim, Y: Dim, Z: Dim>
    Matmul<NamedTensor<B, (Batch, Y, Z)>, NamedTensor<B, (Batch, X, Z)>>
    for NamedTensor<B, (Batch, X, Y)>
{
    fn matmul(self, rhs: NamedTensor<B, (Batch, Y, Z)>) -> NamedTensor<B, (Batch, X, Z)> {
        NamedTensor::from_tensor(self.tensor.matmul(rhs.tensor))
    }
}

impl<B: Backend, Batch1: Dim, Batch2: Dim, X: Dim, Y: Dim, Z: Dim>
    Matmul<NamedTensor<B, (Batch1, Batch2, Y, Z)>, NamedTensor<B, (Batch1, Batch2, X, Z)>>
    for NamedTensor<B, (Batch1, Batch2, X, Y)>
{
    fn matmul(
        self,
        rhs: NamedTensor<B, (Batch1, Batch2, Y, Z)>,
    ) -> NamedTensor<B, (Batch1, Batch2, X, Z)> {
        NamedTensor::from_tensor(self.tensor.matmul(rhs.tensor))
    }
}
