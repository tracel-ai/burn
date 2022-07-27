use crate::tensor::{
    backend::ndarray::{BatchMatrix, NdArrayTensor},
    ops::*,
};
use ndarray::{Dim, Dimension};

impl<P, const D: usize> TensorOpsTranspose<P, D> for NdArrayTensor<P, D>
where
    P: Default + Clone + std::fmt::Debug,
    Dim<[usize; D]>: Dimension,
    Dim<[usize; D]>: Dimension,
{
    fn transpose(&self) -> Self {
        if D > 2 {
            return self.clone();
        }

        let batch_matrix = BatchMatrix::from_ndarray(self.array.clone(), self.shape.clone());

        let arrays = batch_matrix
            .arrays
            .iter()
            .map(|matrix| matrix.t())
            .map(|output| output.into_owned().into_shared())
            .collect();

        let mut shape = self.shape.clone();
        let size0 = shape.dims[D - 2];
        let size1 = shape.dims[D - 1];
        shape.dims[D - 2] = size1;
        shape.dims[D - 1] = size0;

        let output = BatchMatrix::new(arrays, shape.clone());

        Self::from_bmatrix(output)
    }
}
