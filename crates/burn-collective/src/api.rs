use burn_communication::Address;
use burn_tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    CollectiveConfig,
    global::{NodeId, shared::GlobalCollectiveError},
    local_server::get_collective_client,
};

/// Parameters passed to the node for registering on the global level
///
///
/// TODO: More doc on why those things exist
/// TODO: Explain a bit the p2p archi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRegisterParams {
    /// The id of this node, should be unique.
    pub node_id: NodeId,
    /// The address for the connection to the global orchestrator.
    pub global_address: Address,
    /// The address for the connection to this node.
    pub node_address: Address,
    /// The port on which to open the tensor data service for other nodes. Should match the port
    /// given in the node url.
    pub data_service_port: u16,

    /// The number of nodes globally. Should be the same between different nodes
    pub num_nodes: u32,
}

/// Parameters for an all-reduce that should be the same between all devices
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedAllReduceParams {
    pub kind: ReduceKind,
    pub local_strategy: AllReduceStrategy,
    pub global_strategy: Option<AllReduceStrategy>,
}

/// Reduce can be done different ways
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceKind {
    Sum,
    Mean,
}

/// All reduce can be implemented with different algorithms, which all have the same result.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum AllReduceStrategy {
    /// One device is the "central". The other devices, "peripherals", send their tensors to the
    /// central. The central does the reduction, and sends the result back to each peripheral.  
    Centralized,

    /// Devices are organized in a tree structure (with a given arity). Each node reduces its
    /// children's tensors with its own, and sends the result to its parent. Leaf nodes will
    /// simply send their tensors to their parents.
    /// When the root node calculates the result, it is propagated down the tree.
    Tree(u32),

    /// Devices are organized in a ring. The tensors are split into N slices, where N is the
    /// number of devices participating. The slices are progressively sent around the ring until
    /// every device has one fully reduced slice of the tensor. Then, the resulting slices are sent
    /// around until every device has the full result.
    /// See `ring.rs` for details.
    Ring,
}

/// Errors from collective operations
#[allow(unused)]
#[derive(Debug, Clone)]
pub enum CollectiveError {
    /// Cannot un-register a node twice
    MultipleUnregister,
    /// Cannot register a node twice
    MultipleRegister,
    /// Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    /// Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,
    /// Local collective server couldn't respond
    LocalServerMissing,
    /// Another operation was called before Register
    RegisterNotFirstOperation,
    /// The global orchestrator had an error
    Global(GlobalCollectiveError),

    #[allow(unused)]
    Other(String),
}

/// A unique identifier for a device in the context of collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(u32);

impl From<u32> for DeviceId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// Registers a device. `num_devices` must be the same for every register,
/// and `device_id` must be unique.
///
/// With auto-diff backends, make sure to use the inner backend.
pub fn register<B: Backend>(config: &CollectiveConfig) -> Result<(), CollectiveError> {
    let mut client = get_collective_client::<B>();
    client.register(config)
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
pub fn all_reduce<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    config: &CollectiveConfig,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.all_reduce(tensor, config)?;
    let tensor =
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device);

    eprintln!("All-Reduce done");
    Ok(tensor)
}

/// Closes the collective session, unregistering the device
pub fn finish_collective<B: Backend>(config: &CollectiveConfig) -> Result<(), CollectiveError> {
    let client = get_collective_client::<B>();
    client.finish(config.device_id)
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = get_collective_client::<B>();
    client.reset();
}
