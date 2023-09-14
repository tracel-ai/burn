/// Macro to easily generate structs that are a unique ID wrapped in an Arc
#[macro_export]
macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        /// An ID
        pub struct $name {
            id: alloc::sync::Arc<alloc::string::String>,
        }

        impl $name {
            /// Create a new random ID.
            pub fn new() -> Self {
                Self {
                    id: alloc::sync::Arc::new(burn_common::id::IdGenerator::generate()),
                }
            }
        }
    };
}
