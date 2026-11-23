use std::{collections::HashMap, sync::Arc};

use burn_backend::backend::BackendTypes;
use burn_ir::{BackendIr, CustomOpIr, HandleContainer};

/// One member's part of a collective custom operation: its tensors, the operation as it sees it,
/// with its own shards' shapes, and its device.
pub struct CollectiveMember<'a, B: BackendIr> {
    /// The member's tensor handles: the inputs are read from them, and the outputs registered in
    /// them under the operation's ids.
    pub handles: &'a mut HandleContainer<B::Handle>,
    /// The operation with this member's shapes.
    pub op: &'a CustomOpIr,
    /// The member's device.
    pub device: &'a <B as BackendTypes>::Device,
}

/// A handler that runs one custom operation over every member at once, members in order.
pub type CollectiveOpHandler<B> = Arc<dyn Fn(&mut [CollectiveMember<'_, B>]) + Send + Sync>;

/// The handlers a sharded compute device runs for operations planned
/// [collective](super::CustomOpPlan::Collective), keyed by the custom operation's id.
#[derive(Clone)]
pub struct CollectiveOpRegistry<B: BackendIr> {
    handlers: HashMap<String, CollectiveOpHandler<B>>,
}

impl<B: BackendIr> Default for CollectiveOpRegistry<B> {
    fn default() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }
}

impl<B: BackendIr> CollectiveOpRegistry<B> {
    /// A registry without handlers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `handler` for the custom operation `id`, replacing any handler registered for it.
    pub fn register<F>(&mut self, id: &str, handler: F)
    where
        F: Fn(&mut [CollectiveMember<'_, B>]) + Send + Sync + 'static,
    {
        self.handlers.insert(id.into(), Arc::new(handler));
    }

    /// The handler registered for the custom operation `id`.
    pub fn get(&self, id: &str) -> Option<&CollectiveOpHandler<B>> {
        self.handlers.get(id)
    }
}
