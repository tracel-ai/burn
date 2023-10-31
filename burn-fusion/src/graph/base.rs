use super::{FloatOps, TensorId};
use crate::FusionTensor;
use std::{collections::HashMap, sync::Arc};

pub struct FusionServer<B: FusedBackend> {
    candidates: Vec<Box<dyn FusedOps<B::Handle>>>,
    status: Vec<RegisterResult>,
    current_ops: Vec<FloatOps>,
    container: HandleContainer<B::Handle>,
}

impl<B: FusedBackend> FusionServer<B> {
    pub fn register(&mut self, ops: FloatOps) {
        self.current_ops.push(ops.clone());

        for (candidate, status) in self.candidates.iter_mut().zip(self.status.iter_mut()) {
            Self::update_candidate(status, candidate, ops.clone());
        }

        self.maybe_execute();
    }

    pub fn maybe_execute(&mut self) {
        loop {
            let mut num_stoped = 0;

            for status in self.status.iter() {
                match status {
                    RegisterResult::Rejected(_) => num_stoped += 1,
                    _ => {}
                };
            }

            if num_stoped < self.status.len() {
                // not executing, some are still fusing.
                break;
            }

            let mut best_index = None;
            let mut best_score = 0;

            for (i, status) in self.status.iter().enumerate() {
                let properties = match status {
                    RegisterResult::Rejected(properties) => properties,
                    RegisterResult::Accepted(properties) => properties,
                };

                if properties.ready && properties.score >= best_score {
                    best_index = Some(i);
                    best_score = properties.score;
                }
            }

            match best_index {
                Some(index) => self.execute_fused_ops(index),
                None => self.execute_ops(),
            }

            if self.current_ops.is_empty() {
                // No more ops to fuse.
                break;
            }
        }
    }

    fn execute_ops(&mut self) {
        for ops in self.current_ops.drain(..) {
            B::execute_ops(ops, &mut self.container);
        }
    }

    fn execute_fused_ops(&mut self, index: usize) {
        let ops = self.candidates.get_mut(index).unwrap();
        let num_keep = ops.len();
        ops.execute(&mut self.container);

        self.current_ops.drain(0..num_keep);

        for (candidate, status) in self.candidates.iter_mut().zip(self.status.iter_mut()) {
            candidate.reset();
            *status = RegisterResult::Accepted(FusionProperties::default());

            for ops in self.current_ops.iter() {
                Self::update_candidate(status, candidate, ops.clone());
            }
        }
    }

    fn update_candidate(
        status: &mut RegisterResult,
        candidate: &mut Box<dyn FusedOps<B::Handle>>,
        ops: FloatOps,
    ) {
        match status {
            RegisterResult::Rejected(_) => return,
            _ => {}
        };

        *status = candidate.register(ops.clone());
    }
}

pub enum RegisterResult {
    Rejected(FusionProperties),
    Accepted(FusionProperties),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FusionProperties {
    pub score: u64,
    pub ready: bool,
}

pub trait FusedOps<Handle> {
    fn register(&mut self, ops: FloatOps) -> RegisterResult;
    fn execute(&mut self, handles: &mut HandleContainer<Handle>);
    fn reset(&mut self);
    fn len(&self) -> usize;
}

pub trait FusedBackend {
    type Handle;

    fn operations() -> Vec<Box<dyn FusedOps<Self::Handle>>>;
    fn new(shape: Vec<usize>) -> Self::Handle;
    fn execute_ops(ops: FloatOps, handles: &mut HandleContainer<Self::Handle>);
}

pub struct HandleContainer<Handle> {
    handles: HashMap<TensorId, Handle>,
    tensors: HashMap<TensorId, FusionTensor>,
}

pub enum HandleResult<Handle> {
    ReadOnly(Handle),
    ReadWrite(Handle),
}

impl<Handle: Clone> HandleContainer<Handle> {
    fn get(&mut self, id: &TensorId) -> HandleResult<Handle> {
        if let Some(tensor) = self.tensors.get(id) {
            let handle = self.handles.get(&id).unwrap().clone();

            if tensor.can_mut() {
                HandleResult::ReadWrite(handle)
            } else {
                HandleResult::ReadOnly(handle)
            }
        } else {
            panic!("No handle");
        }
    }

    fn new(&mut self, shape: Vec<usize>, handle: Handle) -> FusionTensor {
        let id = TensorId::new();
        let reference = Arc::new(id.clone());
        let tensor = FusionTensor::new(shape, reference);

        self.handles.insert(id.clone(), handle);
        self.tensors.insert(id, tensor.clone());

        tensor
    }
}
