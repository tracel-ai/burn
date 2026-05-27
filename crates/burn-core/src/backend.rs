#[cfg(feature = "ir")]
pub use burn_ir as ir;

pub use burn_backend::*;
pub use burn_backend_extension::backend_extension;

// Dispatch backend extension types
pub use burn_dispatch::{backend::*, device::*, tensor::*};
// Re-export backends (e.g., Cuda)
pub use burn_dispatch::backends::*;
