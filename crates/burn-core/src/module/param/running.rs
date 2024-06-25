use super::ParamId;
use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};

use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use burn_common::stub::Mutex;
use burn_tensor::{
    backend::{AutodiffBackend, Backend},
    Tensor,
};

#[cfg(feature = "std")]
mod threading {
    pub(super) use std::collections::HashMap;
    pub(super) use std::thread::ThreadId;

    #[inline(always)]
    pub(super) fn get_thread_current_id() -> ThreadId {
        std::thread::current().id()
    }
}

#[cfg(not(feature = "std"))]
mod threading {
    pub(super) use burn_common::stub::ThreadId;
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
    value: Arc<Mutex<V>>,
}

// Implement display for the module

impl<V> core::fmt::Display for RunningState<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "RunningState(id={})", self.id)
    }
}

impl<V> ModuleDisplayDefault for RunningState<V> {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add_formatted(&"RunningState".to_string())
            .optional()
    }
}

impl<V> ModuleDisplay for RunningState<V> {}

impl<const D: usize, B: Backend> Module<B> for RunningState<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        let tensor = self.value.lock().unwrap();

        visitor.visit_float(&self.id, &tensor)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let mut tensor = self.value.lock().unwrap();
        let tensor_out = mapper.map_float(&self.id, tensor.clone());

        *tensor = tensor_out;
        core::mem::drop(tensor);

        self
    }

    fn into_record(self) -> Self::Record {
        self.sync();
        let tensor = self.value.lock().unwrap();

        Param::initialized(self.id, tensor.clone())
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        let mut tensor = self.value.lock().unwrap();
        *tensor = record.val().to_device(&tensor.device());
        self.id = record.id;

        core::mem::drop(tensor);

        self
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        let mut tensor = self.value.lock().unwrap();
        let tensor_out = tensor.clone().to_device(device);

        *tensor = tensor_out;
        core::mem::drop(tensor);

        self
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.to_device(device) // Same thing here since no grad.
    }

    fn collect_devices(
        &self,
        mut devices: Vec<<B as Backend>::Device>,
    ) -> Vec<<B as Backend>::Device> {
        let device = self.value.lock().unwrap().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> RunningState<Tensor<B, D>> {
    /// Create a new running state.
    pub fn new(value: Tensor<B, D>) -> Self {
        Self {
            id: ParamId::new(),
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(Mutex::new(value)),
        }
    }

    /// Create a new running state.
    pub fn with_id(id: ParamId, value: Tensor<B, D>) -> Self {
        Self {
            id,
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(Mutex::new(value)),
        }
    }

    /// Create a new running state from a record.
    pub fn from_record(record: Param<Tensor<B, D>>) -> Self {
        let tensor = record.val();
        Self {
            id: record.id,
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(Mutex::new(tensor)),
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
        let value = self.value.lock().unwrap();
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

        let value = self.value.lock().unwrap();
        value.clone()
    }

    fn sync(&self) {
        let mut map = self.values.lock().unwrap();

        if !map.is_empty() {
            self.update_value(&mut map);
        }
    }

    fn update_value(&self, map: &mut HashMap<ThreadId, Tensor<B, D>>) {
        let mut value_updated: Option<Tensor<B, D>> = None;
        let mut counter = 0;

        for (_key, tensor) in map.drain() {
            counter += 1;

            value_updated = match value_updated {
                Some(current) => {
                    let device = current.device();
                    Some(tensor.to_device(&device).add(current))
                }
                None => Some(tensor),
            };
        }

        if let Some(value) = value_updated {
            let value = value.div_scalar(counter);
            let mut value_old = self.value.lock().unwrap();
            *value_old = value;
        }
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for RunningState<Tensor<B, D>> {
    type InnerModule = RunningState<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        self.sync();
        let value = self.value();

        RunningState::with_id(self.id.clone(), value.inner())
    }
}
