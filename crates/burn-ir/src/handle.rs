use hashbrown::HashMap;

use crate::{BackendIr, TensorHandle, TensorId, TensorIr, TensorStatus};

/// Keep all [tensor handles](BackendIr::Handle) in one place and ensure that all resources
/// are used optimally.
pub struct HandleContainer<H> {
    handles: HashMap<TensorId, Handle<H>>,
    counter: u64,
}

// Hand-written perfect derive as we don't require `H: Default`.
impl<H> Default for HandleContainer<H> {
    fn default() -> Self {
        Self {
            handles: HashMap::new(),
            counter: 0,
        }
    }
}

impl<H: Clone> HandleContainer<H> {
    /// Fork the container, useful for autotune.
    pub fn fork(&self) -> Self {
        let mut handles = HashMap::with_capacity(self.handles.len());

        for (id, handle) in self.handles.iter() {
            handles.insert(*id, handle.clone());
        }

        Self {
            handles,
            counter: self.counter,
        }
    }
}

impl<H> core::fmt::Debug for HandleContainer<H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HandleContainer")
            .field("handles", &self.handles.keys()) // only care about the IDs when debugging
            .field("counter", &self.counter)
            .finish()
    }
}

/// Backend [tensor handle](BackendIr::Handle) wrapper tracking their creation state
#[derive(Clone)]
pub enum Handle<H> {
    /// No [tensor handle](BackendIr::Handle) has been created yet
    NotInit,
    /// A [tensor handle](BackendIr::Handle) has been created
    Existing(H),
}

impl<H: Clone> HandleContainer<H> {
    /// Create a new HandleContainer
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            counter: 0,
        }
    }

    /// Register a handle for the given [tensor id](TensorId).
    pub fn register_handle(&mut self, id: TensorId, handle: H) {
        self.handles.insert(id, Handle::Existing(handle));
    }

    /// Whether a handle exists.
    pub fn has_handle(&self, id: &TensorId) -> bool {
        self.handles.contains_key(id)
    }

    /// Get the reference to a handle.
    pub fn get_handle_ref(&self, id: &TensorId) -> Option<&H> {
        self.handles
            .get(id)
            .filter(|h| !matches!(h, Handle::NotInit))
            .map(|h| match h {
                Handle::Existing(handle) => handle,
                Handle::NotInit => unreachable!(),
            })
    }

    /// Get the handle for the given [tensor id](TensorId). The status is used to determine if the
    /// tensor should be popped out of the current tensor map, necessary for inplace operations.
    ///
    /// # Warnings
    ///
    /// Make sure the status corresponds to the operation you want to execute the handle on,
    /// otherwise you might remove a tensor handle that will be required in the future.
    pub fn get_handle(&mut self, id: &TensorId, status: &TensorStatus) -> H {
        let (id, handle) = self
            .handles
            .remove_entry(id)
            .unwrap_or_else(|| panic!("Should have handle for tensor {id:?}"));

        match handle {
            Handle::Existing(handle) => match status {
                TensorStatus::ReadOnly => {
                    self.handles.insert(id, Handle::Existing(handle.clone()));
                    handle
                }
                TensorStatus::ReadWrite => handle,
                TensorStatus::NotInit => panic!(
                    "Cannot get uninitialized tensor {id:?}. Tensor exist but with wrong status"
                ),
            },
            Handle::NotInit => panic!("Cannot get uninitialized handle {id:?}."),
        }
    }

    /// Get the tensor handle for the given [tensor intermediate representation](TensorIr).
    pub fn get_tensor_handle(&mut self, tensor: &TensorIr) -> TensorHandle<H> {
        TensorHandle {
            handle: self.get_handle(&tensor.id, &tensor.status),
            shape: tensor.shape.clone(),
        }
    }

    /// Get the [float tensor](burn_backend::backend::BackendTypes::FloatTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_float_tensor<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::float_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [int tensor](burn_backend::backend::BackendTypes::IntTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_int_tensor<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::int_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [bool tensor](burn_backend::backend::BackendTypes::BoolTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_bool_tensor<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::bool_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [quantized tensor](burn_backend::backend::BackendTypes::QuantizedTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_quantized_tensor<B>(&mut self, tensor: &TensorIr) -> B::QuantizedTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::quantized_tensor(self.get_tensor_handle(tensor))
    }

    /// Register a new [float tensor](burn_backend::backend::BackendTypes::FloatTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_float_tensor<B>(&mut self, id: &TensorId, tensor: B::FloatTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::float_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [quantized tensor](burn_backend::backend::BackendTypes::QuantizedTensorPrimitive) with the corresponding [tensor ids](TensorId).
    pub fn register_quantized_tensor<B>(
        &mut self,
        id: &TensorId,
        tensor: B::QuantizedTensorPrimitive,
    ) where
        B: BackendIr<Handle = H>,
    {
        let handle = B::quantized_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [int tensor](burn_backend::backend::BackendTypes::IntTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_int_tensor<B>(&mut self, id: &TensorId, tensor: B::IntTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::int_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [bool tensor](burn_backend::backend::BackendTypes::BoolTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_bool_tensor<B>(&mut self, id: &TensorId, tensor: B::BoolTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::bool_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Remove tensor handle from container.
    pub fn remove_handle(&mut self, id: TensorId) -> Option<Handle<H>> {
        self.handles.remove(&id)
    }

    /// Remove tensor handle from container if writable
    pub fn free(&mut self, tensor: &TensorIr) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::NotInit => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        };
    }

    /// Returns the number of handles.
    pub fn num_handles(&self) -> usize {
        self.handles.len()
    }

    /// Returns the IDs of all currently registered handles.
    ///
    /// Useful for snapshotting which handles exist at a point in time (e.g., before
    /// executing on a forked context) so that newly registered output handles can
    /// be detected afterwards.
    pub fn handle_ids(&self) -> impl Iterator<Item = &'_ TensorId> {
        self.handles.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorId;

    /// Helper to create a TensorId for tests.
    fn tid(value: u64) -> TensorId {
        TensorId::new(value)
    }

    #[test]
    fn fork_clones_existing_handles() {
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "input_a".to_string());
        container.register_handle(tid(2), "input_b".to_string());

        let fork = container.fork();

        assert_eq!(fork.num_handles(), 2);
        assert!(fork.get_handle_ref(&tid(1)).is_some());
        assert!(fork.get_handle_ref(&tid(2)).is_some());
    }

    #[test]
    fn fork_is_isolated_from_original() {
        // This test documents the core of the autotune clone bug:
        // output handles registered in a fork do NOT appear in the original.
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "input_a".to_string());

        let mut fork = container.fork();

        // Simulate an optimization registering output handles in the fork.
        fork.register_handle(tid(100), "output_x".to_string());
        fork.register_handle(tid(101), "output_y".to_string());

        // The fork has the output handles.
        assert_eq!(fork.num_handles(), 3);
        assert!(fork.get_handle_ref(&tid(100)).is_some());
        assert!(fork.get_handle_ref(&tid(101)).is_some());

        // But the original does NOT — these output handles are lost.
        assert_eq!(container.num_handles(), 1);
        assert!(container.get_handle_ref(&tid(100)).is_none());
        assert!(container.get_handle_ref(&tid(101)).is_none());
    }

    #[test]
    fn fork_mutations_do_not_affect_original() {
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "original_value".to_string());

        let mut fork = container.fork();

        // Overwrite a handle in the fork (e.g., inplace output reuse).
        fork.register_handle(tid(1), "modified_in_fork".to_string());

        // Original is unchanged.
        assert_eq!(
            container.get_handle_ref(&tid(1)),
            Some(&"original_value".to_string())
        );
        assert_eq!(
            fork.get_handle_ref(&tid(1)),
            Some(&"modified_in_fork".to_string())
        );
    }

    #[test]
    fn double_fork_is_fully_isolated() {
        // Simulates what happens when UnsafeTuneContext::get() is called on a Fork:
        // it forks again, creating a second level of isolation.
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "input".to_string());

        let fork1 = container.fork();
        let mut fork2 = fork1.fork();

        fork2.register_handle(tid(200), "deep_output".to_string());

        assert!(fork1.get_handle_ref(&tid(200)).is_none());
        assert!(container.get_handle_ref(&tid(200)).is_none());
        assert!(fork2.get_handle_ref(&tid(200)).is_some());
    }
}
