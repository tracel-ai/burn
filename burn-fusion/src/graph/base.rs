use super::{TensorId, TensorOps};
use crate::FusionTensor;
use burn_tensor::Element;
use std::{collections::HashMap, ops::RangeBounds, rc::Rc, sync::Arc, vec::Drain};

pub struct Graph<B: FusedBackend> {
    operations: Vec<Rc<TensorOps<B::FloatElem, B::IntElem>>>,
}

impl<B: FusedBackend> Graph<B> {
    pub fn add(&mut self, ops: Rc<TensorOps<B::FloatElem, B::IntElem>>) {
        self.operations.push(ops);
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.len() == 0
    }

    pub fn drain<R>(&mut self, range: R) -> Drain<'_, Rc<TensorOps<B::FloatElem, B::IntElem>>>
    where
        R: RangeBounds<usize>,
    {
        self.operations.drain(range)
    }

    pub fn remove<R: RangeBounds<usize>>(&mut self, range: R) {
        self.operations.drain(range);
    }

    pub fn nodes<'a>(&'a self) -> &'a [Rc<TensorOps<B::FloatElem, B::IntElem>>] {
        &self.operations
    }

    pub fn execute_optimization(
        &mut self,
        handles: &mut HandleContainer<B::Handle>,
        index: usize,
        optimizations: &mut [Optimization<B>],
    ) {
        let optimization = optimizations.get_mut(index).unwrap();
        let num_keep = optimization.ops.len();
        optimization.ops.execute(handles);

        self.remove(0..num_keep);

        for optimization in optimizations.iter_mut() {
            optimization.reset();

            for node in self.nodes() {
                optimization.register(node);
            }
        }
    }
    pub fn execute(&mut self, handles: &mut HandleContainer<B::Handle>) {
        for ops in self.drain(..) {
            B::execute_ops(ops, handles);
        }
    }
}

pub struct Optimization<B: FusedBackend> {
    pub ops: Box<dyn FusedOps<B::Handle, B::FloatElem, B::IntElem>>,
    pub status: FusionStatus,
}

impl<B: FusedBackend> Optimization<B> {
    pub fn register(&mut self, ops: &Rc<TensorOps<B::FloatElem, B::IntElem>>) {
        match self.status {
            FusionStatus::Closed(_) => return,
            _ => {}
        };

        self.status = self.ops.register(ops.clone());
    }

    pub fn reset(&mut self) {
        self.ops.reset();
        self.status = FusionStatus::Open(FusionProperties::default());
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
