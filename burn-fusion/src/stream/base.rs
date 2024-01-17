use super::Operation;
use super::OperationConverter;
use super::OperationDescription;
use crate::FusionBackend;

/// A growing list of [tensor operation descriptions](OperationDescription).
pub struct OperationQueue<B: FusionBackend> {
    pub(crate) global: Vec<OperationDescription>,
    pub(crate) relative: Vec<OperationDescription>,
    pub(crate) converter: OperationConverter,
    pub(crate) operations: Vec<Box<dyn Operation<B>>>,
}

impl<B: FusionBackend> Default for OperationQueue<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: FusionBackend> OperationQueue<B> {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            converter: OperationConverter::default(),
            operations: Vec::new(),
        }
    }

    /// Add a new tensor operation to the queue.
    ///
    /// The new [operation description](OperationDescription) will be converted to a local
    /// representation that can be reused when the same pattern emerge in different but similar
    /// scenario, so that the same optimization can be used.
    pub fn add(&mut self, global: OperationDescription, operation: Box<dyn Operation<B>>) {
        let relative = global.to_relative(&mut self.converter);
        self.relative.push(relative);
        self.global.push(global);
        self.operations.push(operation);
    }

    /// The size of the queue.
    pub fn len(&self) -> usize {
        self.global.len()
    }

    /// If the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
