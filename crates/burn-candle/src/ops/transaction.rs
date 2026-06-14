use burn_backend::{
    Backend,
    distributed::DistributedOps,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{
    Candle,
    element::{FloatCandleElement, IntCandleElement},
};

impl TransactionOps<Self> for Candle {}

// DistributedOps has default implementations; Candle does not support collective operations.
impl DistributedOps<Self> for Candle {}
