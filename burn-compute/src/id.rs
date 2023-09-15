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

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Macro to generate id structs with tensor and computing counts
#[macro_export]
macro_rules! memory_id_type {
    ($name:ident) => {
        /// id struct with tensor and computing counts
        pub struct $name {
            id: alloc::sync::Arc<alloc::string::String>,
            computing_count: Arc<core::sync::atomic::AtomicUsize>,
        }

        impl $name {
            pub(crate) fn new() -> Self {
                Self {
                    id: alloc::sync::Arc::new(burn_common::id::IdGenerator::generate()),
                    computing_count: Arc::new(0.into()),
                }
            }

            pub(crate) fn tensor_reference(&self) -> Self {
                Self {
                    id: self.id.clone(),
                    computing_count: self.computing_count.clone(),
                }
            }

            pub(crate) fn compute_reference(&self) -> Self {
                self.computing_count
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Self {
                    id: self.id.clone(),
                    computing_count: self.computing_count.clone(),
                }
            }

            pub(crate) fn get_compute_reference_number(&self) -> usize {
                self.computing_count
                    .load(core::sync::atomic::Ordering::Relaxed)
            }

            pub(crate) fn is_free_to_deallocate(&self) -> bool {
                self.get_compute_reference_number() == 0 && Arc::strong_count(&self.id) <= 1
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl core::hash::Hash for $name {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.id.hash(state);
            }
        }

        impl core::cmp::PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }

        impl core::cmp::Eq for $name {}
    };
}
