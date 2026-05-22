use burn_backend::{
    Backend,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{
    Candle,
    element::{FloatCandleElement, IntCandleElement},
};

impl TransactionOps<Self> for Candle {}
