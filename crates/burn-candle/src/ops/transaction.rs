use burn_tensor::{
    backend::Backend,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{
    element::{FloatCandleElement, IntCandleElement},
    Candle,
};

impl<F: FloatCandleElement, I: IntCandleElement> TransactionOps<Self> for Candle<F, I> {}
