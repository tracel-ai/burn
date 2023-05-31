use burn_tensor::ops::ActivationOps;

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend,
};

impl<G, F, I> ActivationOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
}
