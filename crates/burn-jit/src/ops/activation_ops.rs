use crate::{element::BoolElement, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::ActivationOps;

impl<R, F, I, B> ActivationOps<Self> for JitBackend<R, F, I, B>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
{
}
