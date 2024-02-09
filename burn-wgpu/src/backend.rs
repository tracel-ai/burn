use crate::{codegen::Compiler, compute::WgpuAutotuneKey, tensor::WgpuTensor};
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};
use burn_fusion::FusionDevice;
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
pub struct GpuBackend<R: Runtime> {
    _b: PhantomData<R>,
}

impl<R: Runtime> core::fmt::Debug for GpuBackend<R> {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<R: Runtime> Clone for GpuBackend<R> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<R: Runtime> Default for GpuBackend<R> {
    fn default() -> Self {
        todo!()
    }
}

/// Trait that defines a backend with a Just-In-Time compiler.
pub trait Runtime: Send + Sync + 'static {
    type Compiler: Compiler;
    type Server: ComputeServer<
        Kernel = Box<dyn crate::compute::Kernel>,
        AutotuneKey = WgpuAutotuneKey,
    >;
    type Channel: ComputeChannel<Self::Server>;
    type Device: FusionDevice
        + Default
        + core::hash::Hash
        + PartialEq
        + Eq
        + Clone
        + core::fmt::Debug
        + Sync
        + Send;

    type FullPrecisionBackend: Runtime<
        Compiler = <Self::Compiler as Compiler>::FullPrecisionCompiler,
        Device = Self::Device,
        Server = Self::Server,
        Channel = Self::Channel,
    >;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel>;
}

impl<R: Runtime> Backend for GpuBackend<R> {
    type Device = R::Device;
    type FullPrecisionBackend = GpuBackend<R::FullPrecisionBackend>;

    type FullPrecisionElem = f32;
    type FloatElem = <R::Compiler as Compiler>::Float;
    type IntElem = <R::Compiler as Compiler>::Int;

    type FloatTensorPrimitive<const D: usize> = WgpuTensor<R, Self::FloatElem, D>;
    type IntTensorPrimitive<const D: usize> = WgpuTensor<R, Self::IntElem, D>;
    type BoolTensorPrimitive<const D: usize> = WgpuTensor<R, u32, D>;

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
        let client = R::client(device);
        client.sync();
    }
}
