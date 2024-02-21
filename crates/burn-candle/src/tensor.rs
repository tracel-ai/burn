use std::marker::PhantomData;

use burn_tensor::{Data, Element, Shape};

use crate::{element::CandleElement, CandleDevice};

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

    /// Creates a new tensor from data and a device.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// A new tensor.
    pub fn from_data(data: Data<E, D>, device: CandleDevice) -> Self {
        let candle_shape: candle_core::Shape = (&data.shape.dims).into();
        let tensor =
            candle_core::Tensor::from_slice(data.value.as_slice(), candle_shape, &device.into());
        Self::new(tensor.unwrap())
    }

    pub(crate) fn shape(&self) -> Shape<D> {
        let x: [usize; D] = self.tensor.dims().try_into().unwrap();
        Shape::from(x)
    }
}
