use crate::{
    element::{BoolElement, ByteElement},
    FloatElement, IntElement, JitBackend, JitRuntime,
};
use burn_tensor::ops::ActivationOps;

impl<R, F, I, B, P> ActivationOps<Self> for JitBackend<R, F, I, B, P>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
    P: ByteElement,
{
}
