//! SafeTensors persistence implementation for Burn modules.
//!
//! This module provides efficient tensor serialization/deserialization using the SafeTensors format,
//! with support for filtering, remapping, metadata, and both file and memory-based storage.

mod persister;

pub use persister::{
    MemoryPersister, SafetensorsError, SafetensorsPersister, SafetensorsPersisterConfig,
};

// For backwards compatibility
pub use persister::MemoryPersister as SafetensorsMemoryPersister;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_tests;
