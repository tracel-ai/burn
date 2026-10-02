use core::fmt;

type Source = Box<dyn std::error::Error + Send + Sync>;

/// Why a server could not serve.
#[derive(Debug)]
pub enum ServeError {
    /// The transport could not bind its socket or endpoint, as when the port is taken.
    Bind {
        /// The transport's reason.
        source: Source,
    },
    /// The transport failed while serving, after it had bound.
    Transport {
        /// The transport's reason.
        source: Source,
    },
    /// The server was given no device to host.
    NoDevices,
    /// A device whose backend cannot run remote sessions, such as LibTorch or a remote device.
    UnsupportedDevice {
        /// Which device, and why.
        reason: String,
    },
    /// The devices belong to more than one backend. Every CubeCL runtime is one backend, so CUDA
    /// and wgpu devices can be served together, but not with Flex or NdArray ones.
    MixedBackends,
    /// Custom operations were registered for a backend other than the devices'.
    CustomOpBackend {
        /// The backend they were registered for.
        backend: &'static str,
    },
    /// The Iroh endpoint cannot carry the protocol: another live endpoint has its id.
    InvalidEndpoint {
        /// Why.
        reason: String,
    },
    /// The handlers that stop a blocking `serve` on Ctrl+C or `SIGTERM` could not be installed.
    SignalHandler {
        /// The operating system's reason.
        source: std::io::Error,
    },
}

#[cfg(not(target_family = "wasm"))]
impl ServeError {
    pub(crate) fn bind(source: impl Into<Source>) -> Self {
        Self::Bind {
            source: source.into(),
        }
    }

    pub(crate) fn transport(source: impl Into<Source>) -> Self {
        Self::Transport {
            source: source.into(),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl From<crate::runtime::Interrupted> for ServeError {
    fn from(interrupted: crate::runtime::Interrupted) -> Self {
        Self::transport(interrupted)
    }
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind { source } => write!(f, "cannot bind the server: {source}"),
            Self::Transport { source } => write!(f, "the server's transport failed: {source}"),
            Self::NoDevices => f.write_str("a server needs at least one device"),
            Self::UnsupportedDevice { reason } => write!(f, "cannot serve this device: {reason}"),
            Self::MixedBackends => f.write_str("a server's devices must share one backend"),
            Self::CustomOpBackend { backend } => write!(
                f,
                "custom operations registered for {backend}, which the server's devices do not use"
            ),
            Self::InvalidEndpoint { reason } => {
                write!(f, "cannot serve on this endpoint: {reason}")
            }
            Self::SignalHandler { source } => {
                write!(f, "cannot install the shutdown signal handlers: {source}")
            }
        }
    }
}

impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind { source } | Self::Transport { source } => Some(source.as_ref()),
            Self::SignalHandler { source } => Some(source),
            _ => None,
        }
    }
}
