use burn_tensor::ops::ActivationOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> ActivationOps<CandleBackend<E>> for CandleBackend<E> {}
