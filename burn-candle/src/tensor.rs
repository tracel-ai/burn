use std::marker::PhantomData;

use burn_tensor::Shape;

use crate::element::CandleElement;

/// A reference to a tensor storage.
// pub type StorageRef = Arc<*mut >;

/// A tensor that uses the tch backend.
#[derive(Debug, Clone)]
pub struct CandleTensor<E: CandleElement, const D: usize> {
    pub(crate) tensor: candle_core::Tensor,
    // pub(crate) storage: StorageRef,
    phantom: PhantomData<E>,
}

impl<E: CandleElement, const D: usize> CandleTensor<E, D> {
    /// Create a new tensor.
    pub fn new(tensor: candle_core::Tensor) -> Self {
        // let data = Arc::new(tensor.storage_and_layout());

        Self {
            tensor,
            // storage: data,
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
