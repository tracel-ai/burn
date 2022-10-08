use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    NdArrayElement,
};

impl<P, const D: usize> TensorOpsTranspose<P, D> for NdArrayTensor<P, D>
where
    P: Default + Clone + std::fmt::Debug + NdArrayElement,
{
    fn transpose(&self) -> Self {
        self.swap_dims(D - 2, D - 1)
    }
    fn swap_dims(&self, dim1: usize, dim2: usize) -> Self {
        let mut shape = self.shape;
        let dim1_new = shape.dims[dim2];
        let dim2_new = shape.dims[dim1];

        shape.dims[dim1] = dim1_new;
        shape.dims[dim2] = dim2_new;

        let mut array = self.array.clone();
        array.swap_axes(dim1, dim2);

        Self { array, shape }
    }
}
