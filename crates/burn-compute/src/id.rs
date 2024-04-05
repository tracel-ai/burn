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
pub struct TensorBufRef<Id> {
    id: Arc<Id>,
    all: Arc<()>,
}

#[derive(Clone, Debug)]
pub struct BufRef<Id> {
    id: Id,
    _all: Arc<()>,
}

impl<Id> BufRef<Id>
where
    Id: Clone + core::fmt::Debug,
{
    pub(crate) fn id(&self) -> &Id {
        &self.id
    }
}

impl<Id> TensorBufRef<Id>
where
    Id: Clone + core::fmt::Debug,
{
    pub(crate) fn new(id: Id) -> Self {
        Self {
            id: Arc::new(id),
            all: Arc::new(()),
        }
    }

    pub(crate) fn id(&self) -> &Id {
        &self.id
    }

    pub(crate) fn buf_ref(&self) -> BufRef<Id> {
        BufRef {
            id: self.id.as_ref().clone(),
            _all: self.all.clone(),
        }
    }

    pub(crate) fn can_mut(&self) -> bool {
        // 1 memory management reference with 1 tensor reference.
        Arc::strong_count(&self.id) <= 2
    }

    pub(crate) fn is_free(&self) -> bool {
        // 1 memory management reference with 0 tensor reference.
        Arc::strong_count(&self.id) <= 1
    }

    pub(crate) fn can_be_dealloc(&self) -> bool {
        Arc::strong_count(&self.all) <= 1
    }
}

#[macro_export(local_inner_macros)]
/// Create a new memory ID type.
macro_rules! memory_id_type {
    ($id:ident, $handle_buf_tensor:ident, $handle_buf:ident) => {
        /// Tensor buffer handle.
        #[derive(Clone, Debug)]
        pub struct $handle_buf_tensor {
            value: $crate::id::TensorBufRef<$id>,
        }

        /// Execution buffer handle.
        #[derive(Clone, Debug)]
        pub struct $handle_buf {
            value: $crate::id::BufRef<$id>,
        }

        /// Memory ID.
        #[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
        pub struct $id {
            value: u64,
        }

        impl $handle_buf_tensor {
            /// Create a new ID.
            pub(crate) fn new() -> Self {
                static COUNTER: core::sync::atomic::AtomicU64 =
                    core::sync::atomic::AtomicU64::new(0);

                let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                if value == u64::MAX {
                    core::panic!("Memory ID overflowed");
                }
                Self {
                    value: $crate::id::TensorBufRef::new($id { value }),
                }
            }

            pub(crate) fn handle(&self) -> $handle_buf {
                $handle_buf {
                    value: self.value.buf_ref(),
                }
            }
        }

        impl core::ops::Deref for $handle_buf_tensor {
            type Target = $crate::id::TensorBufRef<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl core::ops::Deref for $handle_buf {
            type Target = $crate::id::BufRef<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl Default for $handle_buf_tensor {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}
