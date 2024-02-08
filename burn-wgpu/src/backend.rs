use crate::{
    codegen::Compiler,
    compute::compute_client,
    element::{FloatElement, IntElement},
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
#[derive(Debug, Default, Clone)]
pub struct GpuBackend<G, C>
where
    G: GraphicsApi,
    C: Compiler,
{
    _g: PhantomData<G>,
    _c: PhantomData<C>,
}

impl<G, C> Backend for GpuBackend<G, C>
where
    G: GraphicsApi + 'static,
    C: Compiler,
    C::Float: FloatElement,
    C::Int: IntElement,
{
    type Device = WgpuDevice;
    type FullPrecisionBackend = GpuBackend<G, C::FullPrecisionCompiler>;

    type FullPrecisionElem = f32;
    type FloatElem = C::Float;
    type IntElem = C::Int;

    type FloatTensorPrimitive<const D: usize> = WgpuTensor<C::Float, D>;
    type IntTensorPrimitive<const D: usize> = WgpuTensor<C::Int, D>;
    type BoolTensorPrimitive<const D: usize> = WgpuTensor<u32, D>;

    fn name() -> String {
        String::from("wgpu")
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }

    fn ad_enabled() -> bool {
        false
    }

    fn sync(device: &Self::Device) {
        let client = compute_client::<G>(device);
        client.sync();
    }
}
