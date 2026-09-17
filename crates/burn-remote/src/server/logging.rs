use tracing_core::Level;
use tracing_subscriber::{
    EnvFilter, Layer, filter::filter_fn, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

/// The `tracing` subscriber a server installs when it owns the process.
///
/// A turnkey server ([`start`](super::RemoteServerBuilder::start)) installs it, so running one is
/// enough to see what its sessions and transfers do. A server composed into an application installs
/// whatever subscriber that application wants: the process-wide one belongs to the program, not to a
/// library it links.
pub struct ServerLogging;

impl ServerLogging {
    /// Install it, with the filter from `RUST_LOG` and `info` by default. Does nothing when a
    /// subscriber is already installed, which is what happens when a process runs several servers.
    pub fn install() {
        let layer = tracing_subscriber::fmt::layer()
            .with_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            // wgpu logs a line per resource at info level, which buries everything else.
            .with_filter(filter_fn(|metadata| {
                let wgpu = metadata
                    .module_path()
                    .is_some_and(|path| path.starts_with("wgpu"));
                !(wgpu && *metadata.level() >= Level::INFO)
            }));

        let _ = registry().with(layer).try_init();
    }
}
