use serde::{Deserialize, Serialize};

/// Unique identifier for any node in the global collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct NodeId(u32);

impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<usize> for NodeId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<i32> for NodeId {
    fn from(value: i32) -> Self {
        Self(value as u32)
    }
}
