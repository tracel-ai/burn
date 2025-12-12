use burn_backend::{
    Backend,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{
    Candle,
    element::{FloatCandleElement, IntCandleElement},
};

impl<F: FloatCandleElement, I: IntCandleElement> TransactionOps<Self> for Candle<F, I> {}
