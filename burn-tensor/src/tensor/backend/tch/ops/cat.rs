use crate::{
    backend::tch::TchKind,
    tensor::{backend::tch::TchTensor, ops::*, Shape},
    TchElement,
};

impl<P: TchElement, const D: usize> TensorOpsCat<P, D> for TchTensor<P, D> {
    fn cat(tensors: Vec<&Self>, dim: usize) -> Self {
        let tensors: Vec<tch::Tensor> = tensors
            .into_iter()
            .map(|t| t.tensor.shallow_clone())
            .collect();
        let tensor = tch::Tensor::cat(&tensors, dim as i64);
        let shape = Shape::from(tensor.size());
        let kind = TchKind::new();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}
