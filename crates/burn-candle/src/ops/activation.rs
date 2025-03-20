use burn_tensor::ops::{ActivationOps, FloatTensor};

use crate::{
    Candle, CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    tensor,
};

impl<F: FloatCandleElement, I: IntCandleElement> ActivationOps<Self> for Candle<F, I> {
    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.gelu().unwrap())
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.relu().unwrap())
    }
}
