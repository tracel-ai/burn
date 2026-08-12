use crate::stream::{OperationConverter, RelativeOps};
use crate::{FusionRuntime, UnfusedOp};
use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr, TensorStatus};

use hashbrown::HashMap;

/// A growing list of [tensor operation descriptions](OperationIr).
///
/// Every queue is associated with a single [`StreamId`](super::super::StreamId) — every tensor it
/// references is local to that stream (cross-stream sharing is handled out-of-band by
/// [`MultiStream::tag_shared_view`](super::super::MultiStream::tag_shared_view), which aliases
/// the source handle under a fresh local tensor id before the op is enqueued).
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
    pub(crate) operations: Vec<UnfusedOp<R>>,
    pub(crate) variables: HashMap<TensorId, TensorStatus>,
    /// Last-use frees of materialized tensors received from another thread,
    /// run at the next execution boundary instead of interrupting the queue
    /// (see [`ReadPlan`](crate::stream::ReadPlan)).
    pub(crate) deferred_frees: Vec<TensorIr>,
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
            deferred_frees: Vec::new(),
        }
    }

    /// Whether any pending operation still references `id`. `variables`
    /// cannot answer this: it keeps entries for tensors whose queued uses
    /// were all `ReadOnly` after those ops executed.
    pub(crate) fn references_tensor(&self, id: TensorId) -> bool {
        self.global
            .iter()
            .any(|op| op.nodes().iter().any(|node| node.id == id))
    }

    /// Free every deferred tensor no pending operation still references,
    /// keeping the rest for the next boundary.
    ///
    /// Called at execution boundaries, right after the referencing ops ran —
    /// the earliest legal point to release the memory.
    pub(crate) fn flush_deferred(&mut self, handles: &mut HandleContainer<R::FusionHandle>) {
        if self.deferred_frees.is_empty() {
            return;
        }
        let deferred = core::mem::take(&mut self.deferred_frees);
        for ir in deferred {
            if self.references_tensor(ir.id) {
                self.deferred_frees.push(ir);
            } else {
                self.variables.remove(&ir.id);
                handles.free(&ir);
            }
        }
    }

    /// Add a new tensor operation to the queue.
    ///
    /// The new [operation intermediate representation](OperationIr) will be converted to a local
    /// representation that can be reused when the same pattern emerge in different but similar
    /// scenario, so that the same optimization can be used.
    pub fn add(&mut self, global: OperationIr, operation: UnfusedOp<R>) {
        for node in global.nodes() {
            self.variables.insert(node.id, node.status);
        }
        let relative = global.to_relative(&mut self.converter);
        self.relative.push(relative);
        self.global.push(global);
        self.operations.push(operation);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use burn_backend::StreamId;

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
