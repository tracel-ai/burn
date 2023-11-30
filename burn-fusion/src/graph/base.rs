use super::Ops;
use super::RelativeGraphConverter;
use super::TensorOpsDescription;
use crate::FusionOps;
use crate::{FusionBackend, HandleContainer};
use std::ops::RangeBounds;

/// The computational graph containing a list of [tensor operation descriptions](TensorOpsDescription).
pub struct Graph<B: FusionBackend> {
    pub(crate) global: Vec<TensorOpsDescription>,
    pub(crate) relative: Vec<TensorOpsDescription>,
    converter: RelativeGraphConverter,
    ops: Vec<Box<dyn Ops<B>>>,
}

impl<B: FusionBackend> Graph<B> {
    pub(crate) fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            converter: RelativeGraphConverter::default(),
            ops: Vec::new(),
        }
    }

    pub(crate) fn lazy_format_relative(
        &self,
    ) -> (&[TensorOpsDescription], Option<&TensorOpsDescription>) {
        let len = self.relative.len();
        if len < 1 {
            return (&self.relative, None);
        }

        (&self.relative[0..len - 1], self.relative.last())
    }

    pub(crate) fn add(&mut self, global: TensorOpsDescription, ops: Box<dyn Ops<B>>) {
        let relative = global.to_relative(&mut self.converter);
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
    ) {
        for ops in self.global.drain(range.clone()) {
            ops.cleanup_tensor(handles)
        }
        self.ops.drain(range);

        // Rebuilt the local graph when removing partially the global graph.
        self.cleanup_relative_graph();

        for node in self.global.iter() {
            let relative = node.to_relative(&mut self.converter);
            self.relative.push(relative);
        }
    }

    fn cleanup_relative_graph(&mut self) {
        self.relative.clear();
        self.converter.clear();
    }

    pub(crate) fn execute_ops(&mut self, handles: &mut HandleContainer<B>, ops: &dyn FusionOps<B>) {
        let num_keep = ops.len();
        let mut context = self.converter.context(handles);
        ops.execute(&mut context);

        self.remove(0..num_keep, handles);
    }

    pub(crate) fn execute(&mut self, handles: &mut HandleContainer<B>) {
        for (description, ops) in self.global.drain(..).zip(self.ops.drain(..)) {
            ops.execute(handles);
            description.cleanup_tensor(handles);
        }
        self.cleanup_relative_graph();
    }
}
