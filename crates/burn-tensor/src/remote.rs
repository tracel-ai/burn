//! Reaching a remote compute server as a client.

#[cfg(not(target_family = "wasm"))]
pub use burn_dispatch::backends::remote::{ConnectError, IrohPeer, IrohPeerBuilder};
pub use burn_dispatch::backends::remote::{
    Endpoint, EndpointAddr, EndpointId, IrohRelays, RelayUrl,
};
