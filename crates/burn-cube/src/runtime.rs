use crate::{codegen::Compiler, compute::CubeTask, ir::Elem};
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

/// Runtime for the CubeCL.
pub trait Runtime: Send + Sync + 'static + core::fmt::Debug {
    /// The compiler used to compile the inner representation into tokens.
    type Compiler: Compiler;
    /// The compute server used to run kernels and perform autotuning.
    type Server: ComputeServer<Kernel = Box<dyn CubeTask<Self::Server>>, FeatureSet = FeatureSet>;
    /// The channel used to communicate with the compute server.
    type Channel: ComputeChannel<Self::Server>;
    /// The device used to retrieve the compute client.
    type Device;

    /// Retrieve the compute client from the runtime device.
    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel>;

    /// The runtime name.
    fn name() -> &'static str;

    /// Return true if global input array lengths should be added to kernel info.
    fn require_array_lengths() -> bool {
        false
    }
}

/// The set of [features](Feature) supported by a [runtime](Runtime).
#[derive(Default)]
pub struct FeatureSet {
    set: alloc::collections::BTreeSet<Feature>,
}

impl FeatureSet {
    pub fn new(features: &[Feature]) -> Self {
        let mut this = Self::default();

        for feature in features {
            this.register(*feature);
        }

        this
    }
    /// Check if the provided [feature](Feature) is supported by the runtime.
    pub fn enabled(&self, feature: Feature) -> bool {
        self.set.contains(&feature)
    }

    /// Register a [feature](Feature) supported by the compute server.
    ///
    /// This should only be used by a [runtime](Runtime) when initializing a device.
    pub fn register(&mut self, feature: Feature) -> bool {
        self.set.insert(feature)
    }
}

/// Every feature that can be supported by a [cube runtime](Runtime).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    /// The subcube feature enables all basic warp/subgroup operations.
    Subcube,
    /// The cmma feature enables cooperative matrix-multiply and accumulate operations.
    Cmma {
        a: Elem,
        b: Elem,
        c: Elem,
        m: u8,
        k: u8,
        n: u8,
    },
}
