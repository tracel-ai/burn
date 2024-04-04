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
pub struct TensorBuffer<Id> {
    id: Arc<Id>,
    execution: Arc<()>,
}

#[derive(Clone, Debug)]
pub struct ExecutionBuffer<Id> {
    id: Id,
    _execution: Arc<()>,
}

impl<Id> ExecutionBuffer<Id>
where
    Id: Clone + core::fmt::Debug,
{
    pub fn id(&self) -> &Id {
        &self.id
    }
}

impl<Id> TensorBuffer<Id>
where
    Id: Clone + core::fmt::Debug,
{
    pub(crate) fn new(id: Id) -> Self {
        Self {
            id: Arc::new(id),
            execution: Arc::new(()),
        }
    }

    pub(crate) fn id(&self) -> &Id {
        &self.id
    }

    pub(crate) fn execution(&self) -> ExecutionBuffer<Id> {
        ExecutionBuffer {
            id: self.id.as_ref().clone(),
            _execution: self.execution.clone(),
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
        // Only memory management reference
        let no_execution_queued = Arc::strong_count(&self.execution) <= 1;
        let no_tensor_holded = Arc::strong_count(&self.id) <= 1;

        no_tensor_holded && no_execution_queued
    }
}

#[macro_export(local_inner_macros)]
/// Create a new memory ID type.
macro_rules! memory_id_type {
    ($id:ident, $handle_buf_tensor:ident, $handle_buf_execution:ident) => {
        /// Tensor buffer handle.
        #[derive(Clone, Debug)]
        pub struct $handle_buf_tensor {
            value: $crate::id::TensorBuffer<$id>,
        }

        /// Execution buffer handle.
        #[derive(Clone, Debug)]
        pub struct $handle_buf_execution {
            value: $crate::id::ExecutionBuffer<$id>,
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
                    value: $crate::id::TensorBuffer::new($id { value }),
                }
            }

            pub(crate) fn execution(&self) -> $handle_buf_execution {
                $handle_buf_execution {
                    value: self.value.execution(),
                }
            }
        }

        impl core::ops::Deref for $handle_buf_tensor {
            type Target = $crate::id::TensorBuffer<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl core::ops::Deref for $handle_buf_execution {
            type Target = $crate::id::ExecutionBuffer<$id>;

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
