use burn_tensor::ops::IntTensorOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> IntTensorOps<CandleBackend<E>> for CandleBackend<E> {}
