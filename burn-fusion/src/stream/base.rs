use super::Ops;
use super::RelativeStreamConverter;
use super::TensorOpsDescription;
use crate::FusionBackend;

/// A growing list of [tensor operation descriptions](TensorOpsDescription).
pub struct Stream<B: FusionBackend> {
    pub(crate) global: Vec<TensorOpsDescription>,
    pub(crate) relative: Vec<TensorOpsDescription>,
    pub(crate) converter: RelativeStreamConverter,
    pub(crate) ops: Vec<Box<dyn Ops<B>>>,
}

impl<B: FusionBackend> Stream<B> {
    pub(crate) fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            converter: RelativeStreamConverter::default(),
            ops: Vec::new(),
        }
    }

    pub(crate) fn add(&mut self, global: TensorOpsDescription, ops: Box<dyn Ops<B>>) {
        let relative = global.to_relative(&mut self.converter);
        self.relative.push(relative);
        self.global.push(global);
        self.ops.push(ops);
    }

    /// The size of the stream.
    pub(crate) fn len(&self) -> usize {
        self.global.len()
    }

    /// If the stream is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
