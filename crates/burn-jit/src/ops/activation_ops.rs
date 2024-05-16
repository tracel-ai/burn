use crate::{FloatElement, IntElement, JitBackend, Runtime};
use burn_tensor::{backend::DeviceOps, ops::ActivationOps};

impl<R, F, I> ActivationOps<Self> for JitBackend<R, F, I>
where
    R: Runtime,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
{
}
