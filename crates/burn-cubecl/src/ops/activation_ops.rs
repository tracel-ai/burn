use crate::{element::BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_tensor::ops::ActivationOps;

impl<R, F, I, BT> ActivationOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}
