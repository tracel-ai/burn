use crate::{
    backend::tch::{TchBackend, TchKind},
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};

impl<E, const D: usize> TensorOpsPrecision<TchBackend<E>, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn to_full_precision(&self) -> TchTensor<f32, D> {
        let kind = TchKind::<f32>::new();
        let tensor = self.tensor.to_kind(kind.kind());
        let shape = self.shape;

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }

    fn from_full_precision(tensor_full: TchTensor<f32, D>) -> TchTensor<E, D> {
        let kind = TchKind::<E>::new();
        let tensor = tensor_full.tensor.to_kind(kind.kind());
        let shape = tensor_full.shape;

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }
}
