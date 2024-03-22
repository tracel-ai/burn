use crate::{FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayTensor};
use burn_tensor::backend::BackendBridge;
use core::marker::PhantomData;

/// Handle precision conversion for the ndarray backend.
pub struct PrecisionBridge<E: FloatNdArrayElement> {
    _e: PhantomData<E>,
}

impl<TargetElement, OriginElement> BackendBridge<NdArray<OriginElement>>
    for PrecisionBridge<TargetElement>
where
    TargetElement: FloatNdArrayElement,
    OriginElement: FloatNdArrayElement,
{
    type Target = NdArray<TargetElement>;

    fn into_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<NdArray<OriginElement>, D>,
        _device: Option<NdArrayDevice>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn from_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self::Target, D>,
        _device: Option<NdArrayDevice>,
    ) -> burn_tensor::ops::FloatTensor<NdArray<OriginElement>, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }
}
