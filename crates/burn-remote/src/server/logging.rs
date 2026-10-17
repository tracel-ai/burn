use tracing_subscriber::{
    EnvFilter, Layer, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

/// The `tracing` subscriber a server installs when it owns the process.
///
/// The blocking [`serve`](super::BackendServer::serve) installs it, so running a server is enough
/// to see what its sessions and transfers do. `serve_async` and `into_protocol` leave the
/// subscriber to the application: the process-wide one belongs to the program, not to a library it
/// links.
pub struct ServerLogging;

impl ServerLogging {
    /// Install it, with the filter from `RUST_LOG`, or `info` with wgpu at `warn` by default. Does
    /// nothing when a subscriber is already installed, which is what happens when a process runs
    /// several servers.
    pub fn install() {
        // wgpu logs a line per resource at info level, which buries everything else.
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,wgpu=warn"));
        let layer = tracing_subscriber::fmt::layer().with_filter(filter);

        let _ = registry().with(layer).try_init();
    }
}
