use burn_tensor::ops::BoolTensorOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> BoolTensorOps<CandleBackend<E>> for CandleBackend<E> {}
