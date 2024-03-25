use crate::{ops::TchOps, LibTorch, TchElement, TchTensor};
use burn_tensor::{backend::BackendBridge, ops::FloatTensor, Device};
use std::marker::PhantomData;

/// Handle precision conversion for the candle backend.
#[derive(Debug)]
pub struct PrecisionBridge<E: TchElement> {
    _e: PhantomData<E>,
}

impl<TElem, OElem> BackendBridge<LibTorch<OElem>> for PrecisionBridge<TElem>
where
    TElem: TchElement,
    OElem: TchElement,
{
    type Target = LibTorch<TElem>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<LibTorch<OElem>, D>,
        device: Option<Device<Self::Target>>,
    ) -> FloatTensor<Self::Target, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(TElem::KIND);

        let tensor = TchTensor::from_existing(tensor, storage);

        if let Some(device) = &device {
            TchOps::to_device(tensor, device)
        } else {
            tensor
        }
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        device: Option<Device<LibTorch<OElem>>>,
    ) -> FloatTensor<LibTorch<OElem>, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(OElem::KIND);

        let tensor = TchTensor::from_existing(tensor, storage);

        if let Some(device) = &device {
            TchOps::to_device(tensor, device)
        } else {
            tensor
        }
    }
}
