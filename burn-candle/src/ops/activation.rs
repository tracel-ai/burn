use burn_tensor::ops::{ActivationOps, FloatTensor};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    tensor, CandleBackend, CandleTensor,
};

impl<F: FloatCandleElement, I: IntCandleElement> ActivationOps<Self> for CandleBackend<F, I> {
    fn gelu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.gelu().unwrap())
    }

    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.relu().unwrap())
    }
}
