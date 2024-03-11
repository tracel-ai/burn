mod base;
mod simple;
mod tune;

/// Contains utilitary for matmul operation
pub mod utils;

pub use base::*;
pub use simple::*;
pub use tune::*;
pub use utils::*;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
pub mod padding;

#[cfg(not(feature = "export_tests"))]
mod padding;

pub mod tiling2d;
pub mod tiling2d_padded;
pub use tiling2d::*;
pub use tiling2d_padded::*;
