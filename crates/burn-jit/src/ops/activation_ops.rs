use crate::{FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::ActivationOps;

impl<R, F, I> ActivationOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
}
