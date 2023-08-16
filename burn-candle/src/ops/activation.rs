use burn_tensor::ops::ActivationOps;

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleBackend,
};

impl<F: FloatCandleElement, I: IntCandleElement> ActivationOps<CandleBackend<F, I>>
    for CandleBackend<F, I>
{
}
