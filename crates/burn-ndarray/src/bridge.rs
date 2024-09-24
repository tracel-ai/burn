use crate::{element::QuantElement, FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayTensor};
use burn_tensor::{backend::BackendBridge, ops::FloatTensor};
use core::marker::PhantomData;

/// Handle precision conversion for the ndarray backend.
#[derive(Debug)]
pub struct PrecisionBridge<E: FloatNdArrayElement> {
    _e: PhantomData<E>,
}

impl<TElem, OElem, QElem> BackendBridge<NdArray<OElem, QElem>> for PrecisionBridge<TElem>
where
    TElem: FloatNdArrayElement,
    OElem: FloatNdArrayElement,
    QElem: QuantElement,
{
    type Target = NdArray<TElem>;

    fn into_target(
        tensor: FloatTensor<NdArray<OElem>>,
        _device: Option<NdArrayDevice>,
    ) -> FloatTensor<Self::Target> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        _device: Option<NdArrayDevice>,
    ) -> FloatTensor<NdArray<OElem>> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }
}
