use super::GraphKey;
use super::LocalGraphConverter;
use super::Ops;
use super::Policy;
use super::TensorOpsDescription;
use crate::FusionOps;
use crate::{FusionBackend, FusionOpsBuilder, FusionProperties, FusionStatus, HandleContainer};
use std::ops::RangeBounds;

/// The computational graph containing a list of [tensor operation descriptions](TensorOpsDescription).
pub struct Graph<B: FusionBackend> {
    pub(crate) global: Vec<TensorOpsDescription>,
    pub(crate) local: Vec<TensorOpsDescription>,
    pub(crate) key: GraphKey,
    converter: LocalGraphConverter,
    ops: Vec<Box<dyn Ops<B>>>,
}

impl<B: FusionBackend> Graph<B> {
    pub(crate) fn new() -> Self {
        Self {
            global: Vec::new(),
            local: Vec::new(),
            key: GraphKey::default(),
            converter: LocalGraphConverter::default(),
            ops: Vec::new(),
        }
    }

    pub(crate) fn key(&self) -> &GraphKey {
        &self.key
    }

    pub(crate) fn add(&mut self, description: TensorOpsDescription, ops: Box<dyn Ops<B>>) {
        let local = description.to_local(&mut self.converter);
        self.key.register(&local);
        self.local.push(local);
        self.global.push(description);
        self.ops.push(ops);
    }

    /// The size of the graph.
    pub fn len(&self) -> usize {
        self.global.len()
    }

    /// If the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.global.len() == 0
    }

    fn remove<R: RangeBounds<usize> + Clone>(
        &mut self,
        range: R,
        handles: &mut HandleContainer<B>,
    ) {
        for ops in self.global.drain(range.clone()) {
            ops.cleanup_tensor(handles)
        }
        self.ops.drain(range);

        // Rebuilt the local graph when removing partially the global graph.
        self.local.clear();
        self.key.clear();
        self.converter.clear();

        for node in self.global.iter() {
            let local = node.to_local(&mut self.converter);
            self.key.register(&local);
            self.local.push(local);
        }
    }

    fn nodes(&self) -> &[TensorOpsDescription] {
        &self.global
    }

    pub(crate) fn execute_ops(
        &mut self,
        handles: &mut HandleContainer<B>,
        optimizations: &mut [Optimization<B>],
        ops: &Box<dyn FusionOps<B>>,
    ) {
        let num_keep = ops.len();
        let mut context = self.converter.context(handles);
        ops.execute(&mut context);

        self.remove(0..num_keep, handles);

        for optimization in optimizations.iter_mut() {
            optimization.reset();

            for node in self.nodes() {
                optimization.register(node);
            }
        }
    }
    pub(crate) fn execute_optimization(
        &mut self,
        handles: &mut HandleContainer<B>,
        index: usize,
        optimizations: &mut [Optimization<B>],
        policy: &mut Policy<Box<dyn FusionOps<B>>>,
    ) {
        let optimization = optimizations.get_mut(index).unwrap();
        let num_keep = optimization.ops.len();

        let mut context = self.converter.context(handles);
        let mut local = Vec::new();
        core::mem::swap(&mut local, &mut self.local);

        let ops = policy.insert(&self.key, &optimization.ops, local, None);
        ops.execute(&mut context);

        self.remove(0..num_keep, handles);

        for optimization in optimizations.iter_mut() {
            optimization.reset();

            for node in self.nodes() {
                optimization.register(node);
            }
        }
    }

    pub(crate) fn execute(&mut self, handles: &mut HandleContainer<B>) {
        for (description, ops) in self.global.drain(..).zip(self.ops.drain(..)) {
            ops.execute(handles);
            description.cleanup_tensor(handles);
        }
    }
}

/// An optimization that can be executed.
#[derive(new)]
pub struct Optimization<B: FusionBackend> {
    /// The [fusion operation](FusionOps) to potentially be executed.
    pub ops: Box<dyn FusionOpsBuilder<B>>,
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
