use core::fmt;

type Source = Box<dyn std::error::Error + Send + Sync>;

/// Why a server could not serve.
#[derive(Debug)]
#[non_exhaustive]
pub enum ServeError {
    /// The transport could not bind its socket or endpoint, as when the port is taken.
    #[non_exhaustive]
    Bind {
        /// The transport's reason.
        source: Source,
    },
    /// The server was given no device to host.
    NoDevices,
    /// A device whose backend cannot run remote sessions, such as LibTorch or a remote device.
    #[non_exhaustive]
    UnsupportedDevice {
        /// Which device, and why.
        reason: String,
    },
    /// The devices belong to more than one backend. Every CubeCL runtime is one backend, so CUDA
    /// and wgpu devices can be served together, but not with Flex or NdArray ones.
    MixedBackends,
    /// Custom operations were registered for a backend other than the devices'.
    #[non_exhaustive]
    CustomOpBackend {
        /// The backend they were registered for.
        backend: &'static str,
    },
    /// The application's Iroh endpoint cannot carry the protocol.
    #[non_exhaustive]
    InvalidEndpoint {
        /// Why.
        reason: String,
    },
    /// The handlers that stop a blocking `serve` on Ctrl+C or `SIGTERM` could not be installed.
    #[non_exhaustive]
    SignalHandler {
        /// The operating system's reason.
        source: std::io::Error,
    },
}

impl ServeError {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn bind(source: impl Into<Source>) -> Self {
        Self::Bind {
            source: source.into(),
        }
    }

    #[doc(hidden)]
    pub fn unsupported_device(reason: impl Into<String>) -> Self {
        Self::UnsupportedDevice {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind { source } => write!(f, "cannot bind the server: {source}"),
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
            Self::Bind { source } => Some(source.as_ref()),
            Self::SignalHandler { source } => Some(source),
            _ => None,
        }
    }
}
