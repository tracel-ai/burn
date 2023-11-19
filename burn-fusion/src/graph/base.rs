use super::TensorOpsDescription;
use crate::{FusionBackend, FusionOps, FusionProperties, FusionStatus, HandleContainer};
use std::{ops::RangeBounds, sync::Arc, vec::Drain};

/// The computational graph containing a list of [tensor operation descriptions](TensorOpsDescription).
pub struct Graph<B: FusionBackend> {
    operations: Vec<Arc<TensorOpsDescription<B>>>,
}

impl<B: FusionBackend> Graph<B> {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
    pub(crate) fn add(&mut self, ops: Arc<TensorOpsDescription<B>>) {
        self.operations.push(ops);
    }

    /// The size of the graph.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// If the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.len() == 0
    }

    fn drain<R>(&mut self, range: R) -> Drain<'_, Arc<TensorOpsDescription<B>>>
    where
        R: RangeBounds<usize>,
    {
        self.operations.drain(range)
    }

    fn remove<R: RangeBounds<usize>>(&mut self, range: R, handles: &mut HandleContainer<B>) {
        for ops in self.operations.drain(range) {
            ops.cleanup_tensor(handles)
        }
    }

    fn nodes(&self) -> &[Arc<TensorOpsDescription<B>>] {
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
        for ops in self.drain(..) {
            ops.execute(handles);
            ops.cleanup_tensor(handles);
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
    pub(crate) fn register(&mut self, ops: &Arc<TensorOpsDescription<B>>) {
        if let FusionStatus::Closed(_) = self.status {
            return;
        }

        self.status = self.ops.register(ops.clone());
    }

    pub(crate) fn reset(&mut self) {
        self.ops.reset();
        self.status = FusionStatus::Open(FusionProperties::default());
    }
}
