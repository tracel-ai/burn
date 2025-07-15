use burn_network::network::NetworkAddress;
use serde::{Deserialize, Serialize};

use crate::{AllReduceParams, AllReduceStrategy, GlobalAllReduceParams, GlobalRegisterParams, ReduceKind, RegisterParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveConfig {
    pub num_devices: u32,
    pub all_reduce_kind: ReduceKind,
    pub local_strategy: AllReduceStrategy,

    // Global parameters (optional)
    pub node_id: Option<u32>,
    pub num_nodes: Option<u32>,
    pub server_address: Option<NetworkAddress>,
    pub client_address: Option<NetworkAddress>,
    pub client_data_port: Option<u16>,
    pub global_strategy: Option<AllReduceStrategy>,
    
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectiveConfig {
    fn new() -> Self {
        Self {
            num_devices: 1,
            all_reduce_kind: ReduceKind::Mean,
            local_strategy: AllReduceStrategy::Tree(2),

            node_id: None,
            num_nodes: None,
            server_address: None,
            client_address: None,
            client_data_port: None,
            global_strategy: None,
        }
    }

    pub fn with_num_devices(mut self, num: u32) -> Self {
        self.num_devices = num;
        self
    }

    pub fn with_all_reduce_kind(mut self, kind: ReduceKind) -> Self {
        self.all_reduce_kind = kind;
        self
    }

    pub fn with_local_strategy(mut self, strategy: AllReduceStrategy) -> Self {
        self.local_strategy = strategy;
        self
    }

    pub fn with_node_id(mut self, id: u32) -> Self {
        self.node_id = Some(id);
        self
    }

    pub fn with_num_nodes(mut self, n: u32) -> Self {
        self.num_nodes = Some(n);
        self
    }

    pub fn with_server_address(mut self, addr: NetworkAddress) -> Self {
        self.server_address = Some(addr);
        self
    }

    pub fn with_client_address(mut self, addr: NetworkAddress) -> Self {
        self.client_address = Some(addr);
        self
    }

    pub fn with_client_data_port(mut self, port: u16) -> Self {
        self.client_data_port = Some(port);
        self
    }

    pub fn with_global_strategy(mut self, strategy: AllReduceStrategy) -> Self {
        self.global_strategy = Some(strategy);
        self
    }

    /// Converts the config into `RegisterParams`, returning an error if only partial global fields are set.
    pub fn register_params(&self) -> Option<RegisterParams> {
        let global_params = match (
            self.node_id,
            self.num_nodes,
            &self.server_address,
            &self.client_address,
            self.client_data_port,
        ) {
            (None, None, None, None, None) => None, // fully local
            (Some(node_id), Some(num_nodes), Some(server_addr), Some(client_addr), Some(port)) => {
                Some(GlobalRegisterParams {
                    node_id,
                    num_nodes,
                    server_address: server_addr.clone(),
                    client_address: client_addr.clone(),
                    client_data_port: port,
                })
            }
            _ => return None,
        };

        Some(RegisterParams {
            num_devices: self.num_devices,
            global_params,
        })
    }

    /// Converts the config into `AllReduceParams`, using optional global strategy.
    pub fn all_reduce_params(&self) -> AllReduceParams {
        AllReduceParams {
            kind: self.all_reduce_kind,
            local_strategy: self.local_strategy,
            global_strategy: self.global_strategy.map(|s| GlobalAllReduceParams {
                strategy: s,
            }),
        }
    }
}
