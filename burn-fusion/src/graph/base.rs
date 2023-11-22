use super::Ops;
use super::TensorOpsDescription;
use crate::{FusionBackend, FusionOps, FusionProperties, FusionStatus, HandleContainer};
use std::{ops::RangeBounds, sync::Arc};

/// The computational graph containing a list of [tensor operation descriptions](TensorOpsDescription).
pub struct Graph<B: FusionBackend> {
    operations: Vec<Arc<TensorOpsDescription>>,
    ops: Vec<Box<dyn Ops<B>>>,
}

impl<B: FusionBackend> Graph<B> {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
            ops: Vec::new(),
        }
    }
    pub(crate) fn add(&mut self, description: Arc<TensorOpsDescription>, ops: Box<dyn Ops<B>>) {
        self.operations.push(description);
        self.ops.push(ops);
    }

    /// The size of the graph.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// If the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.len() == 0
    }

    fn remove<R: RangeBounds<usize> + Clone>(
        &mut self,
        range: R,
        handles: &mut HandleContainer<B>,
    ) {
        for ops in self.operations.drain(range.clone()) {
            ops.cleanup_tensor(handles)
        }
        self.ops.drain(range);
    }

    fn nodes(&self) -> &[Arc<TensorOpsDescription>] {
        &self.operations
    }

    pub(crate) fn execute_optimization(
        &mut self,
        handles: &mut HandleContainer<B>,
        index: usize,
        optimizations: &mut [Optimization<B>],
    ) {
        let optimization = optimizations.get_mut(index).unwrap();
        let num_keep = optimization.ops.len();
        optimization.ops.execute(handles);

        self.remove(0..num_keep, handles);

        for optimization in optimizations.iter_mut() {
            optimization.reset();

            for node in self.nodes() {
                optimization.register(node);
            }
        }
    }

    pub(crate) fn execute(&mut self, handles: &mut HandleContainer<B>) {
        for (description, ops) in self.operations.drain(..).zip(self.ops.drain(..)) {
            ops.execute(handles);
            description.cleanup_tensor(handles);
        }
    }
}

/// An optimization that can be executed.
#[derive(new)]
pub struct Optimization<B: FusionBackend> {
    /// The [fusion operation](FusionOps) to potentially be executed.
    pub ops: Box<dyn FusionOps<B>>,
    /// The current status of the optimization.
    pub status: FusionStatus,
}

impl<B: FusionBackend> Optimization<B> {
    pub(crate) fn register(&mut self, ops: &TensorOpsDescription) {
        if let FusionStatus::Closed(_) = self.status {
            return;
        }

        self.status = self.ops.register(ops);
    }

    pub(crate) fn reset(&mut self) {
        self.ops.reset();
        self.status = FusionStatus::Open(FusionProperties::default());
    }
}
