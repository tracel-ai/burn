use crate::{FloatElement, IntElement, JitBackend, Runtime};
use burn_tensor::ops::ActivationOps;

impl<R: Runtime, F: FloatElement, I: IntElement> ActivationOps<Self> for JitBackend<R, F, I> {}
