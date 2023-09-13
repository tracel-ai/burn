#[macro_export]
macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            id: alloc::sync::Arc<String>,
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    id: alloc::sync::Arc::new(burn_common::id::IdGenerator::generate()),
                }
            }
        }
    };
}
