use crate::{
    element::{FloatElement, IntElement},
    GraphicsApi, WgpuBackend,
};
use burn_tensor::ops::ActivationOps;

impl<G, F, I> ActivationOps<WgpuBackend<G, F, I>> for WgpuBackend<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
}
