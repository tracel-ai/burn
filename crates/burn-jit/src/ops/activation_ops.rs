use crate::{element::BoolElement, FloatElement, IntElement, CubeBackend, JitRuntime};
use burn_tensor::ops::ActivationOps;

impl<R, F, I, BT> ActivationOps<Self> for CubeBackend<R, F, I, BT>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}
