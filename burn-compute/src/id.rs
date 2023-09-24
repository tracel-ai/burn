#[macro_export(local_inner_macros)]
/// Create a new storage ID type.
macro_rules! storage_id_type {
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        /// Storage ID.
        pub struct $name {
            id: alloc::sync::Arc<alloc::string::String>,
        }

        impl $name {
            /// Create a new ID.
            pub fn new() -> Self {
                Self {
                    id: alloc::sync::Arc::new(burn_common::id::IdGenerator::generate()),
                }
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
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        /// Memory ID.
        pub struct $name {
            id: alloc::sync::Arc<alloc::string::String>,
        }

        impl $name {
            /// Create a new ID.
            pub(crate) fn new() -> Self {
                Self {
                    id: alloc::sync::Arc::new(burn_common::id::IdGenerator::generate()),
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}
