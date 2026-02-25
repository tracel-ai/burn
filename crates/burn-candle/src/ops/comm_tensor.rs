use burn_backend::ops::CommunicationTensorOps;

use crate::{Candle, FloatCandleElement, IntCandleElement};

impl<F: FloatCandleElement, I: IntCandleElement> CommunicationTensorOps<Self> for Candle<F, I> {}
