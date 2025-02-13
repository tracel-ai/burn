use crate::{element::BoolElement, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::ActivationOps;

impl<R, F, I, BT> ActivationOps<Self> for JitBackend<R, F, I, BT>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}
