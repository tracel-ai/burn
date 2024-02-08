use crate::{codegen::Compiler, tensor::WgpuTensor, WgpuDevice};
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};
use burn_fusion::FusionDevice;
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
pub struct GpuBackend<B: JitGpuBackend> {
    _b: PhantomData<B>,
}

impl<B: JitGpuBackend> core::fmt::Debug for GpuBackend<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<B: JitGpuBackend> Clone for GpuBackend<B> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<B: JitGpuBackend> Default for GpuBackend<B> {
    fn default() -> Self {
        todo!()
    }
}

pub trait JitGpuBackend: Send + Sync + 'static {
    type FullPrecisionBackend: JitGpuBackend<
        Compiler = <Self::Compiler as Compiler>::FullPrecisionCompiler,
        Device = Self::Device,
    >;
    type Compiler: Compiler;
    type Server: ComputeServer;
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

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        todo!();
    }
}

impl<B: JitGpuBackend> Backend for GpuBackend<B> {
    type Device = B::Device;
    type FullPrecisionBackend = GpuBackend<B::FullPrecisionBackend>;

    type FullPrecisionElem = f32;
    type FloatElem = <B::Compiler as Compiler>::Float;
    type IntElem = <B::Compiler as Compiler>::Int;

    type FloatTensorPrimitive<const D: usize> = WgpuTensor<B, Self::FloatElem, D>;
    type IntTensorPrimitive<const D: usize> = WgpuTensor<B, Self::IntElem, D>;
    type BoolTensorPrimitive<const D: usize> = WgpuTensor<B, u32, D>;

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
        let client = B::client(device);
        client.sync();
    }
}
