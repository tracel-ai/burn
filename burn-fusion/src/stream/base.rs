use super::Ops;
use super::RelativeGraphConverter;
use super::TensorOpsDescription;
use crate::FusionBackend;

/// The computational graph containing a list of [tensor operation descriptions](TensorOpsDescription).
pub struct Stream<B: FusionBackend> {
    pub(crate) global: Vec<TensorOpsDescription>,
    pub(crate) relative: Vec<TensorOpsDescription>,
    pub(crate) converter: RelativeGraphConverter,
    pub(crate) ops: Vec<Box<dyn Ops<B>>>,
}

impl<B: FusionBackend> Stream<B> {
    pub(crate) fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            converter: RelativeGraphConverter::default(),
            ops: Vec::new(),
        }
    }

    pub(crate) fn split_relative_graph(
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
    pub(crate) fn len(&self) -> usize {
        self.global.len()
    }

    /// If the graph is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
