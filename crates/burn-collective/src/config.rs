use std::fmt::Display;

use burn_communication::Address;
use burn_tensor::backend::{AllReduceStrategy, ReduceOperation};
use serde::{Deserialize, Serialize};

/// Parameter struct for setting up and getting parameters for collective operations.
/// Used in most collective api calls.
/// This config is per-node. It is passed to [reduce](crate::register).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveConfig {
    pub(crate) num_devices: usize,
    pub(crate) local_all_reduce_strategy: AllReduceStrategy,
    pub(crate) local_reduce_strategy: ReduceStrategy,
    pub(crate) local_broadcast_strategy: BroadcastStrategy,

    // Global parameters (all are optional, but if one is defined they should all be)
    pub(crate) num_nodes: Option<u32>,
    pub(crate) global_address: Option<Address>,
    pub(crate) node_address: Option<Address>,
    pub(crate) data_service_port: Option<u16>,

    // These strategies may be defined when no other global params are defined
    pub(crate) global_all_reduce_strategy: Option<AllReduceStrategy>,
    pub(crate) global_reduce_strategy: Option<ReduceStrategy>,
    pub(crate) global_broadcast_strategy: Option<BroadcastStrategy>,
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for CollectiveConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_devices = self.num_devices;
        let local_all_reduce_strategy = self.local_all_reduce_strategy;
        let local_reduce_strategy = self.local_reduce_strategy;
        let local_broadcast_strategy = self.local_broadcast_strategy;
        let num_nodes = self.num_nodes;
        let global_address = &self.global_address;
        let node_address = &self.node_address;
        let data_service_port = self.data_service_port;
        let global_all_reduce_strategy = self.global_all_reduce_strategy;
        let global_reduce_strategy = self.global_reduce_strategy;
        let global_broadcast_strategy = self.global_broadcast_strategy;

        write!(
            f,
            r#"
CollectiveConfig {{
    num_devices: {num_devices:?},
    local_all_reduce_strategy: {local_all_reduce_strategy:?},
    local_reduce_strategy: {local_reduce_strategy:?},
    local_broadcast_strategy: {local_broadcast_strategy:?},
    num_nodes: {num_nodes:?},
    global_address: {global_address:?},
    node_address: {node_address:?},
    data_service_port: {data_service_port:?},
    global_all_reduce_strategy: {global_all_reduce_strategy:?},
    global_reduce_strategy: {global_reduce_strategy:?},
    global_broadcast_strategy: {global_broadcast_strategy:?},
}}
"#
        )
    }
}

impl CollectiveConfig {
    fn new() -> Self {
        Self {
            num_devices: 1,
            local_all_reduce_strategy: AllReduceStrategy::Tree(2),
            local_reduce_strategy: ReduceStrategy::Tree(2),
            local_broadcast_strategy: BroadcastStrategy::Tree(2),

            num_nodes: None,
            global_address: None,
            node_address: None,
            data_service_port: None,
            global_all_reduce_strategy: Some(AllReduceStrategy::Ring),
            global_reduce_strategy: Some(ReduceStrategy::Tree(2)),
            global_broadcast_strategy: Some(BroadcastStrategy::Tree(2)),
        }
    }

    /// Selects the number of devices (local peers) on the current node
    pub fn with_num_devices(mut self, num: usize) -> Self {
        self.num_devices = num;
        self
    }

    /// Selects an all-reduce strategy to use on the local level.
    ///
    /// In multi-node contexts, use of the Ring strategy in the local level may be less
    /// advantageous. With multiple nodes, the global all-reduce step is enabled, and its result
    /// is redistributed to all devices.
    /// The Ring strategy inherently distributes the result, which in this context would not be
    /// necessary.
    ///
    /// It is recommended to use a tree strategy locally, and a ring strategy globally.
    pub fn with_local_all_reduce_strategy(mut self, strategy: AllReduceStrategy) -> Self {
        self.local_all_reduce_strategy = strategy;
        self
    }

    /// Selects a reduce strategy to use on the local level.
    pub fn with_local_reduce_strategy(mut self, strategy: ReduceStrategy) -> Self {
        self.local_reduce_strategy = strategy;
        self
    }

    /// Selects a broadcast strategy to use on the local level.
    pub fn with_local_broadcast_strategy(mut self, strategy: BroadcastStrategy) -> Self {
        self.local_broadcast_strategy = strategy;
        self
    }

    /// Set the number of nodes in the collective
    ///
    /// This parameter is a global parameter and should only be set in multi-node contexts
    pub fn with_num_nodes(mut self, n: u32) -> Self {
        self.num_nodes = Some(n);
        self
    }

    /// Set the network address of the Global Collective Orchestrator
    ///  
    /// This parameter is a global parameter and should only be set in multi-node contexts
    pub fn with_global_address(mut self, addr: Address) -> Self {
        self.global_address = Some(addr);
        self
    }

    /// Define the address for this node
    ///
    /// This parameter is a global parameter and should only be set in multi-node contexts
    pub fn with_node_address(mut self, addr: Address) -> Self {
        self.node_address = Some(addr);
        self
    }

    /// Selects the network port on which to expose the tensor data service
    /// used for peer-to-peer tensor downloading.
    ///
    /// This parameter is a global parameter and should only be set in multi-node contexts
    pub fn with_data_service_port(mut self, port: u16) -> Self {
        self.data_service_port = Some(port);
        self
    }

    /// Selects an all-reduce strategy to use on the global level.
    ///
    /// This parameter is a global parameter and should only be set in multi-node contexts.
    /// See [the local strategy](Self::with_local_all_reduce_strategy)
    pub fn with_global_all_reduce_strategy(mut self, strategy: AllReduceStrategy) -> Self {
        self.global_all_reduce_strategy = Some(strategy);
        self
    }

    /// Selects an reduce strategy to use on the global level.
    ///
    /// This parameter is a global parameter and should only be set in multi-node contexts.
    /// See [the local strategy](Self::with_local_reduce_strategy)
    pub fn with_global_reduce_strategy(mut self, strategy: ReduceStrategy) -> Self {
        self.global_reduce_strategy = Some(strategy);
        self
    }

    /// Selects an broadcst strategy to use on the global level.
    ///
    /// This parameter is a global parameter and should only be set in multi-node contexts.
    /// See [the local strategy](Self::with_local_broadcast_strategy)
    pub fn with_global_broadcast_strategy(mut self, strategy: BroadcastStrategy) -> Self {
        self.global_broadcast_strategy = Some(strategy);
        self
    }

    /// Returns whether the config is valid. If only some required global-level parameters are
    /// defined and others are not, the config is invalid.  
    pub fn is_valid(&self) -> bool {
        match (
            self.num_nodes,
            &self.global_address,
            &self.node_address,
            self.data_service_port,
        ) {
            (None, None, None, None) => true,
            (Some(_), Some(_), Some(_), Some(_)) => true,
            // Global parameters have only been partially defined!
            _ => false,
        }
    }

    /// Return the global parameters for registering in a multi-node context.
    ///
    /// If only some global parameters are defined, returns None. Use [is_valid](Self::is_valid) to check for
    /// validity in this case.
    pub(crate) fn global_register_params(&self) -> Option<GlobalRegisterParams> {
        match (
            self.num_nodes,
            &self.global_address,
            &self.node_address,
            self.data_service_port,
        ) {
            // Only local collective
            (None, None, None, None) => None,
            // Local + global collective
            (Some(num_nodes), Some(global_addr), Some(node_addr), Some(data_service_port)) => {
                Some(GlobalRegisterParams {
                    num_nodes,
                    global_address: global_addr.clone(),
                    node_address: node_addr.clone(),
                    data_service_port,
                })
            }
            // Config is invalid!
            _ => None,
        }
    }
}

/// Helper struct for parameters in a multi-node register operation. Either they are all defined,
/// or all not defined. Passed to the global client for registering on the global level and
/// opening the p2p tensor service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRegisterParams {
    /// The address for the connection to the global orchestrator.
    pub global_address: Address,
    /// The address for the connection to this node.
    pub node_address: Address,
    /// The port on which to open the tensor data service for peer-to-peer tensor transfers with
    /// other nodes. Should match the port given in the node url.
    pub data_service_port: u16,

    /// The number of nodes globally. Should be the same between different nodes
    pub num_nodes: u32,
}

/// Parameters for an all-reduce that should be the same between all devices
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedAllReduceParams {
    pub op: ReduceOperation,
    pub local_strategy: AllReduceStrategy,
    pub global_strategy: Option<AllReduceStrategy>,
}

/// Parameters for a reduce that should be the same between all devices
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedReduceParams {}

/// Parameters for a broadcast that should be the same between all devices
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedBroadcastParams {
    pub op: ReduceOperation,
    pub local_strategy: BroadcastStrategy,
    pub global_strategy: Option<BroadcastStrategy>,
}

/// Reduce can be implemented with different algorithms, which all have the same result.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceStrategy {
    /// See [all-reduce](AllReduceStrategy::Centralized)
    Centralized,

    /// See [all-reduce](AllReduceStrategy::Tree)
    Tree(u32),
}

/// Broadcast can be implemented with different algorithms, which all have the same result.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum BroadcastStrategy {
    /// See [all-reduce](AllReduceStrategy::Centralized)
    Centralized,

    /// See [all-reduce](AllReduceStrategy::Tree)
    Tree(u32),
}
