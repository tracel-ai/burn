use super::{TensorId, TensorOps};
use crate::FusionTensor;
use burn_tensor::Element;
use std::{collections::HashMap, rc::Rc, sync::Arc};

pub struct FusionServer<B: FusedBackend> {
    candidates: Vec<Box<dyn FusedOps<B::Handle, B::FloatElem, B::IntElem>>>,
    status: Vec<FusionStatus>,
    current_ops: Vec<Rc<TensorOps<B::FloatElem, B::IntElem>>>,
    handles: HandleContainer<B::Handle>,
}

/// Trait name graph execution strategy.
impl<B: FusedBackend> FusionServer<B> {
    pub fn register(&mut self, ops: TensorOps<B::FloatElem, B::IntElem>) {
        let ops = Rc::new(ops);
        self.current_ops.push(ops.clone());

        for (candidate, status) in self.candidates.iter_mut().zip(self.status.iter_mut()) {
            Self::update_candidate(status, candidate, ops.clone());
        }

        self.maybe_execute();
    }

    pub fn maybe_execute(&mut self) {
        loop {
            let mut num_stopped = 0;

            for status in self.status.iter() {
                match status {
                    FusionStatus::Closed(_) => num_stopped += 1,
                    _ => {}
                };
            }

            if num_stopped < self.status.len() {
                // not executing, some are still fusing.
                break;
            }

            let mut best_index = None;
            let mut best_score = 0;

            for (i, status) in self.status.iter().enumerate() {
                let properties = match status {
                    FusionStatus::Closed(properties) => properties,
                    FusionStatus::Open(properties) => properties,
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
            B::execute_ops(ops, &mut self.handles);
        }
    }

    fn execute_fused_ops(&mut self, index: usize) {
        let ops = self.candidates.get_mut(index).unwrap();
        let num_keep = ops.len();
        ops.execute(&mut self.handles);

        self.current_ops.drain(0..num_keep);

        for (candidate, status) in self.candidates.iter_mut().zip(self.status.iter_mut()) {
            candidate.reset();
            *status = FusionStatus::Open(FusionProperties::default());

            for ops in self.current_ops.iter() {
                Self::update_candidate(status, candidate, ops.clone());
            }
        }
    }

    fn update_candidate(
        status: &mut FusionStatus,
        candidate: &mut Box<dyn FusedOps<B::Handle, B::FloatElem, B::IntElem>>,
        ops: Rc<TensorOps<B::FloatElem, B::IntElem>>,
    ) {
        match status {
            FusionStatus::Closed(_) => return,
            _ => {}
        };

        *status = candidate.register(ops.clone());
    }
}

pub enum FusionStatus {
    /// No more operation can be fused.
    Closed(FusionProperties),
    /// More operations can be fused.
    Open(FusionProperties),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FusionProperties {
    pub score: u64,
    pub ready: bool,
}

pub trait FusedOps<Handle, FloatElem, IntElem>
where
    FloatElem: Element,
    IntElem: Element,
{
    fn register(&mut self, ops: Rc<TensorOps<FloatElem, IntElem>>) -> FusionStatus;
    fn execute(&mut self, handles: &mut HandleContainer<Handle>);
    fn reset(&mut self);
    fn len(&self) -> usize;
}

pub trait FusedBackend {
    type Handle;
    type FloatElem: Element;
    type IntElem: Element;

    fn operations() -> Vec<Box<dyn FusedOps<Self::Handle, Self::FloatElem, Self::IntElem>>>;
    fn new(shape: Vec<usize>) -> Self::Handle;
    fn execute_ops(
        ops: Rc<TensorOps<Self::FloatElem, Self::IntElem>>,
        handles: &mut HandleContainer<Self::Handle>,
    );
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
