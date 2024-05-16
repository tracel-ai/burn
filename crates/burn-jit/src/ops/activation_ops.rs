use crate::{FloatElement, IntElement, JitBackend, JitRuntime, Runtime};
use burn_tensor::{backend::DeviceOps, ops::ActivationOps};

impl<R, F, I> ActivationOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
}
