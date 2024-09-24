use crate::{
    element::{FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};
use burn_tensor::{backend::BackendBridge, ops::FloatTensor, Device};
use std::marker::PhantomData;

/// Handle precision conversion for the candle backend.
#[derive(Debug)]
pub struct PrecisionBridge<E: FloatCandleElement> {
    _e: PhantomData<E>,
}

impl<TElem, OElem, IntElem> BackendBridge<Candle<OElem, IntElem>> for PrecisionBridge<TElem>
where
    TElem: FloatCandleElement,
    OElem: FloatCandleElement,
    IntElem: IntCandleElement,
{
    type Target = Candle<TElem, IntElem>;

    fn into_target(
        tensor: FloatTensor<Candle<OElem, IntElem>>,
        device: Option<Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        CandleTensor::new(tensor.tensor.to_dtype(TElem::DTYPE).unwrap())
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        device: Option<Device<Candle<OElem, IntElem>>>,
    ) -> FloatTensor<Candle<OElem, IntElem>> {
        CandleTensor::new(tensor.tensor.to_dtype(OElem::DTYPE).unwrap())
    }
}
