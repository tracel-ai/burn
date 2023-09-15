#[macro_export(local_inner_macros)]
macro_rules! storage_id_type {
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            id: alloc::sync::Arc<alloc::string::String>,
        }

        impl $name {
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
macro_rules! memory_id_type {
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            id: alloc::sync::Arc<alloc::string::String>,
        }

        impl $name {
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
