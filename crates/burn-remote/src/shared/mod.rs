mod task;

#[allow(unused_imports)]
pub(crate) use task::*;

/// We define the communication protocol here
pub(crate) type RemoteProtocol = burn_communication::websocket::WebSocket;
