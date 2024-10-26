use burn_tensor::ops::{ActivationOps, FloatTensor};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    tensor, Candle, CandleTensor,
};

impl<F: FloatCandleElement, I: IntCandleElement> ActivationOps<Self> for Candle<F, I> {
    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.gelu().unwrap())
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.relu().unwrap())
    }
}
