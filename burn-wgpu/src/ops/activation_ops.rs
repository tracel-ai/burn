use crate::{codegen::Compiler, GpuBackend, GraphicsApi};
use burn_tensor::ops::ActivationOps;

impl<G, C> ActivationOps<GpuBackend<G, C>> for GpuBackend<G, C>
where
    G: GraphicsApi + 'static,
    C: Compiler,
{
}
