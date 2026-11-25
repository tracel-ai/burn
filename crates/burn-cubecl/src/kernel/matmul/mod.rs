mod base;
mod tile;
mod tune;

/// Contains utilities for matmul operation
pub mod utils;

pub use base::*;
#[cfg(any(feature = "fusion", test))]
pub use tune::fastest_routine;
pub use tune::{MatmulAutotuneGroups, m_grouped, routed};
pub use utils::*;
