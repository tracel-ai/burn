use crate::{codegen::Compiler, compute::Kernel};
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

/// Runtime for the [just-in-time backend](crate::JitBackend).
pub trait Runtime: Send + Sync + 'static + core::fmt::Debug {
    /// The compiler used to compile the inner representation into tokens.
    type Compiler: Compiler;
    /// The compute server used to run kernels and perform autotuning.
    type Server: ComputeServer<Kernel = Kernel>;
    /// The channel used to communicate with the compute server.
    type Channel: ComputeChannel<Self::Server>;
    /// The device used to retrieve the compute client.
    type Device: DeviceOps;

    /// Retrieve the compute client from the runtime device.
    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel>;

    /// The runtime name.
    fn name() -> &'static str;

    /// Return true if global input array lengths should be added to kernel info.
    fn require_array_lengths() -> bool {
        false
    }
}

/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// The handle device trait allows to get an id for a backend device.
pub trait DeviceOps:
    Clone + Default + PartialEq + Eq + core::hash::Hash + Send + Sync + core::fmt::Debug
{
    /// Return the [device id](DeviceId).
    fn id(&self) -> DeviceId;
}
