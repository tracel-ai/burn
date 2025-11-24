use std::sync::Arc;

use crate::FusionRuntime;
use crate::stream::{OperationConverter, OperationStreams, RelativeOps, execution::Operation};
use burn_ir::{OperationIr, TensorId, TensorStatus};
use burn_std::id::StreamId;

use hashbrown::HashMap;

/// A growing list of [tensor operation descriptions](OperationIr).
pub struct OperationQueue<R: FusionRuntime> {
    /// List of operation descriptions. These contain the exact tensor IDs
    /// and shapes so that kernels can be run correctly.
    ///
    /// The length of this list is the same as the length of the `operations` list.
    pub(crate) global: Vec<OperationIr>,
    /// List of operation descriptions. The tensor IDs and shapes are relative
    /// because we don't need to know the exact values, but they are sufficient to
    /// determine which operations can be fused.
    pub(crate) relative: Vec<OperationIr>,
    pub(crate) converter: OperationConverter,
    pub(crate) operations: Vec<Arc<dyn Operation<R>>>,
    pub(crate) variables: HashMap<TensorId, (StreamId, TensorStatus)>,
}

impl<R: FusionRuntime> Default for OperationQueue<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: FusionRuntime> OperationQueue<R> {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            converter: OperationConverter::default(),
            operations: Vec::new(),
            variables: HashMap::new(),
        }
    }

    /// Add a new tensor operation to the queue.
    ///
    /// The new [operation intermediate representation](OperationIr) will be converted to a local
    /// representation that can be reused when the same pattern emerge in different but similar
    /// scenario, so that the same optimization can be used.
    pub fn add(
        &mut self,
        global: OperationIr,
        operation: Arc<dyn Operation<R>>,
        streams: &OperationStreams,
        current: StreamId,
    ) {
        for node in global.nodes() {
            if let Some(stream_id) = streams.get(node.id) {
                self.variables.insert(node.id, (stream_id, node.status));
            } else {
                self.variables.insert(node.id, (current, node.status));
            }
        }
        let relative = global.to_relative(&mut self.converter);
        self.relative.push(relative);
        self.global.push(global);
        self.operations.push(operation);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn stream_id_from_different_threads() {
        let current = StreamId::current();

        let thread1 = std::thread::spawn(|| (StreamId::current(), StreamId::current()));
        let thread2 = std::thread::spawn(StreamId::current);

        let (stream_1, stream_11) = thread1.join().unwrap();
        let stream_2 = thread2.join().unwrap();

        assert_ne!(current, stream_1, "Should be different from thread 1");
        assert_ne!(current, stream_2, "Should be different from thread 2");
        assert_ne!(
            stream_1, stream_2,
            "Should be different from different threads"
        );
        assert_eq!(
            stream_1, stream_11,
            "Should be the same, since same thread."
        );
    }
}
