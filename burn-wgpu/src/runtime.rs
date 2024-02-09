use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};
use burn_fusion::FusionDevice;

use crate::{codegen::Compiler, compute::JitAutotuneKey};

/// Just-In-Time runtime.
pub trait Runtime: Send + Sync + 'static {
    type Compiler: Compiler;
    type Server: ComputeServer<
        Kernel = Box<dyn crate::compute::Kernel>,
        AutotuneKey = JitAutotuneKey,
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

    type FullPrecisionRuntime: Runtime<
        Compiler = <Self::Compiler as Compiler>::FullPrecisionCompiler,
        Device = Self::Device,
        Server = Self::Server,
        Channel = Self::Channel,
    >;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel>;
}
