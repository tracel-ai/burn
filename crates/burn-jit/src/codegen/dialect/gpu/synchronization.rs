use serde::{Deserialize, Serialize};

/// All synchronization types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Synchronization {
    // A workgroup barrier
    WorkgroupBarrier,
}
