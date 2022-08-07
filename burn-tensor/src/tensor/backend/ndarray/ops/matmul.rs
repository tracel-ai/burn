use crate::tensor::{
    backend::ndarray::{BatchMatrix, NdArrayTensor},
    ops::*,
};
use ndarray::LinalgScalar;

impl<P, const D: usize> TensorOpsMatmul<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug,
{
    fn matmul(&self, other: &Self) -> Self {
        let batch_self = BatchMatrix::from_ndarray(self.array.clone(), self.shape.clone());
        let batch_other = BatchMatrix::from_ndarray(other.array.clone(), other.shape.clone());

        let self_iter = batch_self.arrays.iter();
        let other_iter = batch_other.arrays.iter();
        let arrays = self_iter
            .zip(other_iter)
            .map(|(lhs, rhs)| lhs.dot(rhs))
            .map(|output| output.into_shared())
            .collect();

        let mut shape = self.shape.clone();
        shape.dims[D - 1] = other.shape.dims[D - 1];
        let output = BatchMatrix::new(arrays, shape.clone());

        Self::from_bmatrix(output)
    }
}
