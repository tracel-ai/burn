use super::TensorOps;
use crate::HandleContainer;
use burn_tensor::{backend::Backend, Element};
use std::{ops::RangeBounds, rc::Rc, vec::Drain};

pub struct Graph<B: FusedBackend> {
    operations: Vec<Rc<TensorOps<B::FloatElem, B::IntElem>>>,
}

impl<B: FusedBackend> Graph<B> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
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
        handles: &mut HandleContainer<B>,
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

    pub fn execute(&mut self, handles: &mut HandleContainer<B>) {
        for ops in self.drain(..) {
            B::execute_ops(ops, handles);
        }
    }
}

#[derive(new)]
pub struct Optimization<B: FusedBackend> {
    pub ops: Box<dyn FusedOps<B>>,
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

pub trait FusedOps<B: FusedBackend> {
    fn register(&mut self, ops: Rc<TensorOps<B::FloatElem, B::IntElem>>) -> FusionStatus;
    fn execute(&mut self, handles: &mut HandleContainer<B>);
    fn reset(&mut self);
    fn len(&self) -> usize;
}

pub trait FusedBackend: Backend {
    type Handle: Clone;

    fn operations() -> Vec<Box<dyn FusedOps<Self>>>;
    fn new(shape: Vec<usize>) -> Self::Handle;
    fn execute_ops(
        ops: Rc<TensorOps<Self::FloatElem, Self::IntElem>>,
        handles: &mut HandleContainer<Self>,
    );
}
