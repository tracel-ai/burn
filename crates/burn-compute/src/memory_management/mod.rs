pub(crate) mod memory_pool;

mod base;

pub use base::*;

/// Dynamic memory management strategy.
pub mod dynamic;
/// Simple memory management strategy.
pub mod simple;
