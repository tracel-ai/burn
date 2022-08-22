use crate::{
    backend::ndarray::NdArrayBackend,
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    ElementConversion, NdArrayElement,
};

impl<E, const D: usize> TensorOpsPrecision<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn to_full_precision(&self) -> NdArrayTensor<f32, D> {
        let array = self.array.mapv(|a| a.to_elem()).into_shared();

        NdArrayTensor {
            shape: self.shape,
            array,
        }
    }

    fn from_full_precision(tensor_full: NdArrayTensor<f32, D>) -> NdArrayTensor<E, D> {
        let array = tensor_full.array.mapv(|a| a.to_elem()).into_shared();

        NdArrayTensor {
            shape: tensor_full.shape,
            array,
        }
    }
}
