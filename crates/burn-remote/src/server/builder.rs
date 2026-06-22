use burn_backend::tensor::Device;
use burn_communication::websocket::{WebSocket, WsServer};
use burn_ir::{BackendIr, CustomOpIr, HandleContainer};
use burn_router::CustomOpRegistry;

use super::base::RemoteServer;

/// Transport used to serve remote clients.
///
/// Selects the protocol the [`RemoteServerBuilder`] serves on. Only WebSocket exists today; the
/// enum keeps the protocol an open axis.
#[derive(Debug, Clone, Copy)]
pub enum Channel {
    /// WebSocket server bound to `0.0.0.0:port`.
    WebSocket {
        /// Port to bind on.
        port: u16,
    },
}

/// Default port used when none is configured on the builder.
const DEFAULT_PORT: u16 = 3000;

/// Builder for a remote-execution server.
///
/// Configures the transport ([`channel`](Self::channel) / [`port`](Self::port)) and the custom
/// operation handlers ([`custom_op`](Self::custom_op) / [`custom_ops`](Self::custom_ops)), then
/// starts the server with [`start`](Self::start) (blocking) or [`start_async`](Self::start_async).
///
/// The builder is generic over the concrete backend `B`: custom ops are typed by `B`, since their
/// handlers call into `B`'s primitives. A backend extension hosts its ops here — the server-side
/// counterpart of the client building `OperationIr::Custom`.
///
/// ```rust,ignore
/// RemoteServerBuilder::new(devices)
///     .port(3000)
///     .custom_op("fused_matmul_add_relu", |handles, ir| {
///         let ([lhs, rhs, bias], [out]) = ir.as_fixed::<3, 1>();
///         let lhs = handles.get_float_tensor::<MyBackend>(lhs);
///         let rhs = handles.get_float_tensor::<MyBackend>(rhs);
///         let bias = handles.get_float_tensor::<MyBackend>(bias);
///         let result = <MyBackend as MyExt>::fused_matmul_add_relu(lhs, rhs, bias);
///         handles.register_float_tensor::<MyBackend>(&out.id, result);
///     })
///     .start();
/// ```
pub struct RemoteServerBuilder<B: BackendIr> {
    devices: Vec<Device<B>>,
    channel: Channel,
    custom_ops: CustomOpRegistry<B>,
}

impl<B: BackendIr> RemoteServerBuilder<B> {
    /// Create a new builder hosting the given devices.
    ///
    /// `devices` is indexed by the device index a client selects at session init; `devices[0]` is
    /// the default device. Must be non-empty. Defaults to WebSocket on port `3000` with no custom
    /// ops.
    pub fn new(devices: Vec<Device<B>>) -> Self {
        Self {
            devices,
            channel: Channel::WebSocket { port: DEFAULT_PORT },
            custom_ops: CustomOpRegistry::default(),
        }
    }

    /// Select the transport (protocol and its configuration).
    pub fn channel(mut self, channel: Channel) -> Self {
        self.channel = channel;
        self
    }

    /// Set the port, keeping the WebSocket transport. Convenience over [`channel`](Self::channel).
    pub fn port(mut self, port: u16) -> Self {
        self.channel = Channel::WebSocket { port };
        self
    }

    /// Register a handler for a [custom operation](burn_ir::OperationIr::Custom), keyed by `id`.
    ///
    /// The `id` must match the one the client puts in its [`CustomOpIr`]. Chainable; registering
    /// the same id twice keeps the last handler.
    pub fn custom_op<F>(mut self, id: &str, handler: F) -> Self
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr) + Send + Sync + 'static,
    {
        self.custom_ops.register(id, handler);
        self
    }

    /// Replace the custom-op handlers with a prebuilt [`CustomOpRegistry`].
    pub fn custom_ops(mut self, custom_ops: CustomOpRegistry<B>) -> Self {
        self.custom_ops = custom_ops;
        self
    }

    /// Start the server on the caller's async runtime, serving until shutdown.
    pub async fn start_async(self) {
        match self.channel {
            Channel::WebSocket { port } => {
                let server = WsServer::new(port);
                RemoteServer::<B, WebSocket>::start(self.devices, server, self.custom_ops).await;
            }
        }
    }

    /// Start the server, blocking the current thread until shutdown.
    pub fn start(self) {
        let runtime = tokio::runtime::Runtime::new().expect("Failed to build the tokio runtime");
        runtime.block_on(self.start_async());
    }
}
