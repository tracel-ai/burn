use crate::{
    compute::compute_client,
    element::{FloatElement, IntElement},
    tensor::WgpuTensor,
    AutoGraphicsApi, GraphicsApi, WgpuDevice,
};
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
#[derive(Debug, Default, Clone)]
pub struct WgpuBackend<G = AutoGraphicsApi, F = f32, I = i32>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<G: GraphicsApi + 'static, F: FloatElement, I: IntElement> Backend for WgpuBackend<G, F, I> {
    type Device = WgpuDevice;
    type FullPrecisionBackend = WgpuBackend<G, f32, i32>;

    type FullPrecisionElem = f32;
    type FloatElem = F;
    type IntElem = I;

    type FloatTensorPrimitive<const D: usize> = WgpuTensor<F, D>;
    type IntTensorPrimitive<const D: usize> = WgpuTensor<I, D>;
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
