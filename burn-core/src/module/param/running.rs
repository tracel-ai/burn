use alloc::{string::ToString, sync::Arc, vec, vec::Vec};

use super::{load_with_id, state_with_id, ParamId};
use crate::module::{ADModule, LoadingError, Module, ModuleMapper, ModuleVisitor, Param, State};
use burn_tensor::{
    backend::{ADBackend, Backend},
    Data, Tensor,
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
    values: Arc<Mutex<HashMap<ThreadId, V>>>,
    value: Arc<RwLock<V>>,
}

impl<B: Backend, const D: usize> From<RunningState<Tensor<B, D>>>
    for Param<RunningState<Tensor<B, D>>>
{
    fn from(value: RunningState<Tensor<B, D>>) -> Self {
        Param {
            id: ParamId::new(),
            value,
        }
    }
}

impl<const D: usize, B: Backend> Module for Param<RunningState<Tensor<B, D>>> {
    type Backend = B;

    fn num_params(&self) -> usize {
        let tensor = self.value.value.read().unwrap();
        tensor.shape().num_elements()
    }

    fn devices(&self) -> Vec<B::Device> {
        let tensor = self.value.value.read().unwrap();
        vec![tensor.device()]
    }

    fn to_device(self, device: &B::Device) -> Self {
        self.value.sync();

        let mut tensor = self.value.value.write().unwrap();
        tensor.inplace(|tensor| tensor.to_device(device));
        core::mem::drop(tensor);

        self
    }

    fn state(&self) -> State<B::FloatElem> {
        self.sync();

        let tensor = self.value.value.read().unwrap();
        let state = State::Data(tensor.to_data().serialize());

        state_with_id(self.id.clone(), state)
    }

    fn load(mut self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.sync();

        match state {
            State::Data(data) => {
                let mut tensor = self.value.value.write().unwrap();
                *tensor = Tensor::from_data_device(Data::from(data), &tensor.device())
            }
            _ => return Err(LoadingError::new("Can't load tensor".to_string())),
        };

        self.id = id.clone();
        Ok(self)
    }

    fn detach(self) -> Self {
        self.sync();

        let mut tensor = self.value.value.write().unwrap();
        tensor.inplace(|tensor| tensor.detach());
        core::mem::drop(tensor);

        self
    }

    fn visit<V: ModuleVisitor<Self::Backend>>(&self, visitor: &mut V) {
        let tensor = self.value.value.read().unwrap();

        visitor.visit(&self.id, &tensor)
    }

    fn map<M: ModuleMapper<Self::Backend>>(self, mapper: &mut M) -> Self {
        let mut tensor = self.value.value.write().unwrap();
        let tensor_out = mapper.map(&self.id, tensor.clone());

        *tensor = tensor_out;
        core::mem::drop(tensor);

        self
    }
}

impl<const D: usize, B: Backend> RunningState<Tensor<B, D>> {
    /// Create a new running state.
    pub fn new(value: Tensor<B, D>) -> Self {
        Self {
            values: Arc::new(Mutex::new(HashMap::new())),
            value: Arc::new(RwLock::new(value)),
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
    /// register their update before the actual synchonization needs to happen.
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

impl<const D: usize, B: ADBackend> ADModule for Param<RunningState<Tensor<B, D>>> {
    type ADBackend = B;

    type InnerModule = Param<RunningState<Tensor<B::InnerBackend, D>>>;

    fn inner(self) -> Self::InnerModule {
        self.sync();
        let value = self.value.value();

        Param {
            id: self.id,
            value: RunningState::new(value.inner()),
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module.sync();

        let value = module.value.value();

        Param {
            id: module.id,
            value: RunningState::new(Tensor::from_inner(value)),
        }
    }
}
