use burn_tensor::ops::ModuleOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> ModuleOps<CandleBackend<E>> for CandleBackend<E> {}
