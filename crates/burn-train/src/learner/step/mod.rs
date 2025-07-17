/// The trainer module.
pub mod train;

/// Module for a trainer that uses collective operations
#[cfg(feature = "collective")]
pub mod collective;
