use crate::memory_id_type;
use crate::memory_management::{MemoryBinding, MemoryHandle};

// The ChunkId allows to keep track of how many references there are to a specific chunk.
memory_id_type!(ChunkId, ChunkHandle);
// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId, SliceHandle, SliceBinding);

/// A tensor memory handle, referring to either a chunk or a slice.
#[derive(Debug, Clone)]
pub struct MemoryPoolHandle {
    pub slice: SliceHandle,
}

/// Binding of the [dynamic handle](DynamicHandle).
#[derive(Debug, Clone)]
pub struct MemoryPoolBinding {
    pub slice: SliceBinding,
}

impl MemoryBinding for MemoryPoolBinding {}

impl MemoryHandle<MemoryPoolBinding> for MemoryPoolHandle {
    fn can_mut(&self) -> bool {
        self.slice.can_mut()
    }

    fn binding(self) -> MemoryPoolBinding {
        MemoryPoolBinding {
            slice: self.slice.binding(),
        }
    }
}
