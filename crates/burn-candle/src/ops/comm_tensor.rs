use burn_backend::{
    ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
};

use crate::{Candle, FloatCandleElement, IntCandleElement};

impl<F: FloatCandleElement, I: IntCandleElement> CommunicationTensorOps<Self> for Candle<F, I> {}
