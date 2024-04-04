#[macro_export(local_inner_macros)]
/// Create a new storage ID type.
macro_rules! storage_id_type {
    ($name:ident) => {
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

        impl From<u64> for $name {
            fn from(value: u64) -> Self {
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

#[macro_export(local_inner_macros)]
/// Create a new memory ID type.
macro_rules! memory_id_type {
    ($id:ident, $handle:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq, Debug)]
        /// Memory handle.
        pub struct $handle {
            id: alloc::sync::Arc<$id>,
        }

        #[derive(Clone, Hash, PartialEq, Eq, Debug)]
        /// Memory ID.
        pub struct $id {
            value: u64,
        }

        impl $handle {
            /// Create a new ID.
            pub(crate) fn new() -> Self {
                use core::sync::atomic::{AtomicU64, Ordering};

                static COUNTER: AtomicU64 = AtomicU64::new(0);

                let value = COUNTER.fetch_add(1, Ordering::Relaxed);
                if value == u64::MAX {
                    core::panic!("Memory ID overflowed");
                }
                let id = $id { value };

                Self {
                    id: alloc::sync::Arc::new(id),
                }
            }
            /// Get the ID from the handle.
            pub fn id(&self) -> $id {
                self.id.as_ref().clone()
            }
        }

        impl Default for $handle {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}
