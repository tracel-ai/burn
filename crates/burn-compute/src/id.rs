use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use alloc::sync::Arc;

#[macro_export(local_inner_macros)]
/// Create a new storage ID type.
macro_rules! storage_id_type {
    ($name:ident) => {
        /// Storage ID.
        #[derive(Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            value: u64,
        }

        impl $name {
            /// Create a new ID.
            pub fn new() -> Self {
                use core::sync::atomic::{AtomicU64, Ordering};

                static COUNTER: AtomicU64 = AtomicU64::new(0);

                let value = COUNTER.fetch_add(1, Ordering::Relaxed);
                if value == u64::MAX {
                    core::panic!("Memory ID overflowed");
                }
                Self { value }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryHandle {
    pub(crate) id: u64,
    execution_count: Arc<AtomicU32>,
}

impl MemoryHandle {
    pub(crate) fn new(id: u64) -> Self {
        Self {
            id,
            execution_count: Arc::new(0.into()),
        }
    }

    pub(crate) fn start_execution(&self) -> u64 {
        AtomicU32::fetch_add(&self.execution_count, 1, Ordering::Relaxed);
        self.id
    }

    pub(crate) fn end_execution(&self) {
        AtomicU32::fetch_sub(&self.execution_count, 1, Ordering::Relaxed);
    }

    pub(crate) fn can_mut(&self, limit_ref: usize) -> bool {
        Arc::strong_count(&self.execution_count) <= limit_ref
    }

    pub(crate) fn is_free(&self) -> bool {
        let count = AtomicU32::load(&self.execution_count, Ordering::Relaxed);
        Arc::strong_count(&self.execution_count) <= 1 && count == 0
    }
}

#[macro_export(local_inner_macros)]
/// Create a new memory ID type.
macro_rules! memory_id_type {
    ($id:ident, $handle:ident) => {
        #[derive(Clone, Debug)]
        /// Memory handle.
        pub struct $handle {
            id: $crate::id::MemoryHandle,
        }

        #[derive(Clone, Hash, PartialEq, Eq, Debug)]
        /// Memory ID.
        pub struct $id {
            value: u64,
        }

        impl $handle {
            /// Create a new ID.
            pub(crate) fn new() -> Self {
                static COUNTER: core::sync::atomic::AtomicU64 =
                    core::sync::atomic::AtomicU64::new(0);

                let id = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                if id == u64::MAX {
                    core::panic!("Memory ID overflowed");
                }
                Self {
                    id: $crate::id::MemoryHandle::new(id),
                }
            }
            /// Get the ID from the handle.
            pub fn id(&self) -> $id {
                $id {
                    value: self.id.id.clone(),
                }
            }

            pub(crate) fn start_execution(&self) -> $id {
                $id { value: self.id.start_execution() }
            }

            pub(crate) fn end_execution(&self) {
                self.id.end_execution()
            }

            pub(crate) fn can_mut(&self, limit_ref: usize) -> bool {
                self.id.can_mut(limit_ref)
            }

            pub(crate) fn is_free(&self) -> bool {
                self.id.is_free()
            }
        }

        impl Default for $handle {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}
