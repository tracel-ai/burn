use alloc::sync::Arc;

#[macro_export(local_inner_macros)]
/// Create a new storage ID type.
macro_rules! storage_id_type {
    ($name:ident) => {
        /// Storage ID.
        #[derive(Clone, Hash, PartialEq, Eq, Debug)]
        pub struct $name {
            value: usize,
        }

        impl $name {
            /// Create a new ID.
            pub fn new() -> Self {
                use core::sync::atomic::{AtomicUsize, Ordering};

                static COUNTER: AtomicUsize = AtomicUsize::new(0);

                let value = COUNTER.fetch_add(1, Ordering::Relaxed);
                if value == usize::MAX {
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

/// Reference to a buffer handle.
#[derive(Clone, Debug)]
pub struct HandleRef<Id> {
    id: Arc<Id>,
    all: Arc<()>,
}

/// Reference to buffer binding.
#[derive(Clone, Debug)]
pub struct BindingRef<Id> {
    id: Id,
    _all: Arc<()>,
}

impl<Id> BindingRef<Id>
where
    Id: Clone + core::fmt::Debug,
{
    /// The id associated to the buffer.
    pub(crate) fn id(&self) -> &Id {
        &self.id
    }
}

impl<Id> HandleRef<Id>
where
    Id: Clone + core::fmt::Debug,
{
    /// Create a new handle.
    pub(crate) fn new(id: Id) -> Self {
        Self {
            id: Arc::new(id),
            all: Arc::new(()),
        }
    }

    /// The id associated to the handle.
    pub(crate) fn id(&self) -> &Id {
        &self.id
    }

    /// Get the binding.
    pub(crate) fn binding(self) -> BindingRef<Id> {
        BindingRef {
            id: self.id.as_ref().clone(),
            _all: self.all,
        }
    }

    /// If the handle can be mut.
    pub(crate) fn can_mut(&self) -> bool {
        // 1 memory management reference with 1 tensor reference.
        Arc::strong_count(&self.id) <= 2
    }

    /// If the resource is free.
    pub(crate) fn is_free(&self) -> bool {
        Arc::strong_count(&self.all) <= 1
    }
}

#[macro_export(local_inner_macros)]
/// Create new memory ID types.
macro_rules! memory_id_type {
    ($id:ident, $handle:ident) => {
        /// Memory Handle.
        #[derive(Clone, Debug)]
        pub struct $handle {
            value: $crate::id::HandleRef<$id>,
        }

        /// Memory ID.
        #[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
        pub struct $id {
            pub(crate) value: usize,
        }

        impl $handle {
            /// Create a new ID.
            pub(crate) fn new() -> Self {
                let value = Self::gen_id();
                Self {
                    value: $crate::id::HandleRef::new($id { value }),
                }
            }

            fn gen_id() -> usize {
                static COUNTER: core::sync::atomic::AtomicUsize =
                    core::sync::atomic::AtomicUsize::new(0);

                let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                if value == usize::MAX {
                    core::panic!("Memory ID overflowed");
                }

                value
            }
        }

        impl core::ops::Deref for $handle {
            type Target = $crate::id::HandleRef<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl Default for $handle {
            fn default() -> Self {
                Self::new()
            }
        }
    };

    ($id:ident, $handle:ident, $binding:ident) => {
        memory_id_type!($id, $handle);

        /// Binding of a memory handle.
        #[derive(Clone, Debug)]
        pub struct $binding {
            value: $crate::id::BindingRef<$id>,
        }

        impl $handle {
            pub(crate) fn binding(self) -> $binding {
                $binding {
                    value: self.value.binding(),
                }
            }
        }

        impl core::ops::Deref for $binding {
            type Target = $crate::id::BindingRef<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }
    };
}
