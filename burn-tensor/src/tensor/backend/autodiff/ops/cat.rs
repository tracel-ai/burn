use crate::tensor::backend::backend::Backend;
use crate::tensor::{backend::autodiff::ADTensor, ops::*};

impl<B: Backend, const D: usize> TensorOpsCat<B::Elem, D> for ADTensor<D, B> {
    fn cat(tensors: Vec<&Self>, dim: usize) -> Self {
        let tensors_inner: Vec<B::TensorPrimitive<D>> =
            tensors.into_iter().map(|a| a.tensor()).collect();
        let tensors_inner_ref: Vec<&B::TensorPrimitive<D>> = tensors_inner.iter().collect();

        let tensor = TensorOpsCat::cat(tensors_inner_ref, dim);

        Self::from_tensor(tensor)
    }
}
