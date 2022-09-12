use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    NdArrayElement,
};
use ndarray::{Axis, IxDyn};

impl<P: NdArrayElement, const D: usize> TensorOpsCat<P, D> for NdArrayTensor<P, D> {
    fn cat(tensors: Vec<&Self>, dim: usize) -> Self {
        let mut shape = *tensors.get(0).unwrap().shape();
        shape.dims[dim] = tensors.len();

        let arrays: Vec<ndarray::ArrayView<P, IxDyn>> =
            tensors.into_iter().map(|t| t.array.view()).collect();
        let array = ndarray::concatenate(Axis(dim), &arrays)
            .unwrap()
            .into_shared();

        Self { array, shape }
    }
}
