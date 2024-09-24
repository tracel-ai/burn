use crate::{ops::TchOps, LibTorch, QuantElement, TchElement, TchTensor};
use burn_tensor::{backend::BackendBridge, ops::FloatTensor, Device};
use std::marker::PhantomData;

/// Handle precision conversion for the candle backend.
#[derive(Debug)]
pub struct PrecisionBridge<E: TchElement> {
    _e: PhantomData<E>,
}

impl<TElem, OElem, QElem> BackendBridge<LibTorch<OElem, QElem>> for PrecisionBridge<TElem>
where
    TElem: TchElement,
    OElem: TchElement,
    QElem: QuantElement,
{
    type Target = LibTorch<TElem>;

    fn into_target(
        tensor: FloatTensor<LibTorch<OElem>>,
        device: Option<Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(TElem::KIND);

        let tensor = TchTensor::from_existing(tensor, storage);

        if let Some(device) = &device {
            TchOps::to_device(tensor, device)
        } else {
            tensor
        }
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        device: Option<Device<LibTorch<OElem>>>,
    ) -> FloatTensor<LibTorch<OElem>> {
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
