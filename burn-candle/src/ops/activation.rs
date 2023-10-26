use burn_tensor::ops::{ActivationOps, FloatTensor};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    tensor, Candle, CandleTensor,
};

impl<F: FloatCandleElement, I: IntCandleElement> ActivationOps<Self> for Candle<F, I> {
    fn gelu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.gelu().unwrap())
    }

    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.relu().unwrap())
    }
}
