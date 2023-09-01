use burn_tensor::ops::ActivationOps;

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    tensor, CandleBackend, CandleTensor,
};

use super::base::FloatTensor;

impl<F: FloatCandleElement, I: IntCandleElement> ActivationOps<CandleBackend<F, I>>
    for CandleBackend<F, I>
{
    fn gelu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.gelu().unwrap())
    }

    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.relu().unwrap())
    }
}
