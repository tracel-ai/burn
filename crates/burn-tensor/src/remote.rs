//! Reaching a remote compute server as a client.

pub use burn_dispatch::backends::remote::{
    ConnectError, Endpoint, EndpointAddr, EndpointId, IrohRelays, RelayUrl,
};
#[cfg(not(target_family = "wasm"))]
pub use burn_dispatch::backends::remote::{IrohPeer, IrohPeerBuilder};
