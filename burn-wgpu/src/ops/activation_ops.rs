use crate::{
    element::{FloatElement, IntElement},
    GraphicsApi, Wgpu,
};
use burn_tensor::ops::ActivationOps;

impl<G, F, I> ActivationOps<Wgpu<G, F, I>> for Wgpu<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
}
