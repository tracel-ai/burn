use burn_backend::{ops::ActivationOps, tensor::FloatTensor};

use crate::{
    Candle, CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    tensor,
};

impl ActivationOps<Self> for Candle {
    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.gelu().unwrap())
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.relu().unwrap())
    }
}
