// TODO: integrate w/ dispatch
// #[cfg(feature = "remote")]
// pub use burn_remote as remote;
// #[cfg(feature = "remote")]
// pub use burn_remote::RemoteBackend;

// #[cfg(feature = "router")]
// pub use burn_router::Router;
// #[cfg(feature = "router")]
// pub use burn_router as router;

#[cfg(feature = "ir")]
pub use burn_ir as ir;

pub use burn_backend::*;
pub use burn_backend_extension::backend_extension;

// Dispatch backend extension types
pub use burn_dispatch::{backend::*, device::*, tensor::*};
// Re-export backends (e.g., Cuda)
pub use burn_dispatch::backends::*;
