use alloc::sync::Arc;

use super::ParamId;
use crate::module::{ADModule, Module, ModuleMapper, ModuleVisitor, Param};
use burn_tensor::{
    backend::{ADBackend, Backend},
    Tensor,
};

#[cfg(feature = "std")]
mod threading {
    pub(super) use std::collections::HashMap;
    pub(super) use std::sync::{Mutex, RwLock};
    pub(super) use std::thread::ThreadId;

    #[inline(always)]
    pub(super) fn get_thread_current_id() -> ThreadId {
        std::thread::current().id()
    }
}

#[cfg(not(feature = "std"))]
mod threading {
    pub(super) use burn_common::stub::{Mutex, RwLock, ThreadId};
    pub(super) use hashbrown::HashMap;

    #[inline(always)]
    pub(super) fn get_thread_current_id() -> ThreadId {
        panic!("Current thread id is not available")
    }
}

// Re-export items from the disabled/enabled blocks
use threading::*;

/// A state that can be updated during the forward pass while being thread safe.
///
/// # Note
///
/// The state value is the average of all updates on all threads.
#[derive(Clone, Debug)]
pub struct RunningState<V> {
    id: ParamId,
    values: Arc<Mutex<HashMap<ThreadId, V>>>,
    value: Arc<RwLock<V>>,
}

impl<const D: usize, B: Backend> Module<B> for RunningState<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        let tensor = self.value.read().unwrap();

        visitor.visit(&self.id, &tensor)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let mut tensor = self.value.write().unwrap();
        let tensor_out = mapper.map(&self.id, tensor.clone());

        *tensor = tensor_out;
        core::mem::drop(tensor);

        self
    }

    fn into_record(self) -> Self::Record {
        self.sync();
        let tensor = self.value.read().unwrap();

        Param::new(self.id, tensor.clone())
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        let mut tensor = self.value.write().unwrap();
        *tensor = record.value.to_device(&tensor.device());
        self.id = record.id;

        core::mem::drop(tensor);

        self
    }
}

impl<const D: usize, B: Backend> RunningState<Tensor<B, D>> {
    /// Create a new running state.
    pub fn new(value: Tensor<B, D>) -> Self {
        Self {
            id: ParamId::new(),
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(RwLock::new(value)),
        }
    }

    /// Create a new running state.
    pub fn with_id(id: ParamId, value: Tensor<B, D>) -> Self {
        Self {
            id,
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(RwLock::new(value)),
        }
    }

    /// Create a new running state from a record.
    pub fn from_record(record: Param<Tensor<B, D>>) -> Self {
        Self {
            id: record.id,
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(RwLock::new(record.value)),
        }
    }

    /// Update the value on the current thread.
    pub fn update(&self, value: Tensor<B, D>) {
        let thread_id = get_thread_current_id();
        let mut map = self.values.lock().unwrap();

        if map.contains_key(&thread_id) {
            self.update_value(&mut map);
        }

        map.insert(thread_id, value);
    }

    /// Get the current value,
    ///
    /// # Note
    ///
    /// The current value might be outdated by one update.
    pub fn value(&self) -> Tensor<B, D> {
        let value = self.value.read().unwrap();
        value.clone()
    }

    /// Get the current value and make sure it is sync.
    ///
    /// # Note
    ///
    /// Don't use this function after an update on the same thread where other threads might have to
    /// register their update before the actual synchronization needs to happen.
    pub fn value_sync(&self) -> Tensor<B, D> {
        let thread_id = get_thread_current_id();
        let mut map = self.values.lock().unwrap();

        if map.contains_key(&thread_id) {
            self.update_value(&mut map);
        }

        let value = self.value.read().unwrap();
        value.clone()
    }

    fn sync(&self) {
        let mut map = self.values.lock().unwrap();

        if !map.is_empty() {
            self.update_value(&mut map);
        }
    }

    fn update_value(&self, map: &mut HashMap<ThreadId, Tensor<B, D>>) {
        let mut value_updated = None;
        let mut counter = 0;

        for (_key, tensor) in map.drain() {
            counter += 1;

            value_updated = match value_updated {
                Some(current) => Some(tensor.add(current)),
                None => Some(tensor),
            };
        }

        if let Some(value) = value_updated {
            let value = value.div_scalar(counter);
            let mut value_old = self.value.write().unwrap();
            *value_old = value;
        }
    }
}

impl<const D: usize, B: ADBackend> ADModule<B> for RunningState<Tensor<B, D>> {
    type InnerModule = RunningState<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        self.sync();
        let value = self.value();

        RunningState::with_id(self.id.clone(), value.inner())
    }
}
