use super::EndCondition;
use super::Ops;
use super::OptimizationPath;
use super::Policy;
use super::RelativeGraphConverter;
use super::TensorOpsDescription;
use crate::FusionOps;
use crate::{FusionBackend, FusionOpsBuilder, FusionProperties, FusionStatus, HandleContainer};
use std::ops::RangeBounds;

/// The computational graph containing a list of [tensor operation descriptions](TensorOpsDescription).
pub struct Graph<B: FusionBackend> {
    pub(crate) global: Vec<TensorOpsDescription>,
    pub(crate) relative: Vec<TensorOpsDescription>,
    pub(crate) key: OptimizationPath,
    converter: RelativeGraphConverter,
    ops: Vec<Box<dyn Ops<B>>>,
}

impl<B: FusionBackend> Graph<B> {
    pub(crate) fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            key: OptimizationPath::default(),
            converter: RelativeGraphConverter::default(),
            ops: Vec::new(),
        }
    }

    pub(crate) fn to_relative(&mut self, global: &TensorOpsDescription) -> TensorOpsDescription {
        global.to_relative(&mut self.converter)
    }

    pub(crate) fn add(
        &mut self,
        global: TensorOpsDescription,
        relative: TensorOpsDescription,
        ops: Box<dyn Ops<B>>,
    ) {
        self.relative.push(relative);
        self.global.push(global);
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
        policy: Option<&mut Policy<Box<dyn FusionOps<B>>>>,
    ) {
        for ops in self.global.drain(range.clone()) {
            ops.cleanup_tensor(handles)
        }
        self.ops.drain(range);

        // Rebuilt the local graph when removing partially the global graph.
        self.cleanup_relative_graph();

        if let Some(policy) = policy {
            println!("Rebuilt the state");
            for node in self.global.iter() {
                let relative = node.to_relative(&mut self.converter);
                self.relative.push(relative);
                policy.action(&mut self.key, &self.relative, EndCondition::Forced);
            }
        }
    }

    fn cleanup_relative_graph(&mut self) {
        self.relative.clear();
        self.key.clear();
        self.converter.clear();
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

        self.remove(0..num_keep, handles, None);

        for optimization in optimizations.iter_mut() {
            optimization.reset();
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
        let mut relative = Vec::new();
        core::mem::swap(&mut relative, &mut self.relative);
        let mut next_ops = relative.split_off(num_keep);
        let next_ops = match next_ops.is_empty() {
            true => None,
            false => Some(next_ops.remove(0)),
        };

        let ops = policy.register_new(&mut self.key, &optimization.ops, relative, next_ops);
        ops.execute(&mut context);

        self.remove(0..num_keep, handles, Some(policy));

        for optimization in optimizations.iter_mut() {
            optimization.reset();

            for node in self.relative.iter() {
                optimization.register(node);
            }
        }
    }

    pub(crate) fn execute(&mut self, handles: &mut HandleContainer<B>) {
        for (description, ops) in self.global.drain(..).zip(self.ops.drain(..)) {
            ops.execute(handles);
            description.cleanup_tensor(handles);
        }
        self.cleanup_relative_graph();
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
