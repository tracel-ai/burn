use burn_tensor::ops::TensorOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> TensorOps<CandleBackend<E>> for CandleBackend<E> {}
