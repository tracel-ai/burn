use serde::{Deserialize, Serialize};

/// All synchronization types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Synchronization {
    // Synchronizize units in a cube.
    SyncUnits,
}
