use std::marker::PhantomData;

use burn_tensor::Shape;

use crate::element::CandleElement;

/// A tensor that uses the candle backend.
#[derive(Debug, Clone)]
pub struct CandleTensor<E: CandleElement, const D: usize> {
    pub(crate) tensor: candle_core::Tensor,
    phantom: PhantomData<E>,
}

impl<E: CandleElement, const D: usize> CandleTensor<E, D> {
    /// Create a new tensor.
    pub fn new(tensor: candle_core::Tensor) -> Self {
        Self {
            tensor,
            phantom: PhantomData,
        }
    }
}

impl<E: CandleElement, const D: usize> CandleTensor<E, D> {
    pub(crate) fn shape(&self) -> Shape<D> {
        let x: [usize; D] = self.tensor.shape().dims().try_into().unwrap();
        Shape::from(x)
    }
}
