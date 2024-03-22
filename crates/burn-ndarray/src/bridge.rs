use crate::{FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayTensor};
use burn_tensor::{backend::BackendBridge, ops::FloatTensor};
use core::marker::PhantomData;

/// Handle precision conversion for the ndarray backend.
#[derive(Debug)]
pub struct PrecisionBridge<E: FloatNdArrayElement> {
    _e: PhantomData<E>,
}

impl<TElem, OElem> BackendBridge<NdArray<OElem>> for PrecisionBridge<TElem>
where
    TElem: FloatNdArrayElement,
    OElem: FloatNdArrayElement,
{
    type Target = NdArray<TElem>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<NdArray<OElem>, D>,
        _device: Option<NdArrayDevice>,
    ) -> FloatTensor<Self::Target, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        _device: Option<NdArrayDevice>,
    ) -> FloatTensor<NdArray<OElem>, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }
}
