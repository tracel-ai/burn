use core::marker::PhantomData;

use burn_tensor::backend::BackendBridge;

use crate::{FloatNdArrayElement, NdArray, NdArrayTensor};

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
        _device: burn_tensor::Device<Self::Target>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn from_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self::Target, D>,
        _device: burn_tensor::Device<NdArray<OriginElement>>,
    ) -> burn_tensor::ops::FloatTensor<NdArray<OriginElement>, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }
}
