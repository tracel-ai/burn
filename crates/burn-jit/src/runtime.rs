use crate::{
    codegen::Compiler,
    compute::{JitAutotuneKey, Kernel},
};
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

/// Runtime for the [just-in-time backend](crate::JitBackend).
pub trait Runtime: Send + Sync + 'static + core::fmt::Debug {
    /// The compiler used to compile the inner representation into tokens.
    type Compiler: Compiler;
    /// The compute server used to run kernels and perform autotuning.
    type Server: ComputeServer<Kernel = Kernel, AutotuneKey = JitAutotuneKey>;
    /// The channel used to communicate with the compute server.
    type Channel: ComputeChannel<Self::Server>;
    /// The device used to retrieve the compute client.
    type Device: burn_tensor::backend::DeviceOps
        + Default
        + core::hash::Hash
        + PartialEq
        + Eq
        + Clone
        + core::fmt::Debug
        + Sync
        + Send;

    /// Retrieve the compute client from the runtime device.
    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel>;

    /// The runtime name.
    fn name() -> &'static str;

    /// Return true if global input array lengths should be added to kernel info.
    fn require_array_lengths() -> bool {
        false
    }
}
