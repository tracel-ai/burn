use crate::{element::FloatNdArrayElement, NdArrayBackend};
use burn_tensor::ops::ActivationOps;

impl<E: FloatNdArrayElement> ActivationOps<NdArrayBackend<E>> for NdArrayBackend<E> {}
