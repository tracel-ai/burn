//! Outgoing task buffering.

use crate::shared::RemoteMessage;

/// Accumulates outgoing [`RemoteMessage`]s on the runner thread and decides when a batch is ready
/// for the wire.
///
/// Pure buffering logic — no runtime, no socket. [`push`](Self::push) reports when the
/// flush threshold is reached so the caller can flush; [`take`](Self::take) drains the
/// buffer into a frame, leaving it empty for the next batch.
///
/// Batching is wire-level only: every task keeps its own `StreamId`/`RequestId`, so the
/// server sees the same per-task semantics whether tasks arrive batched or one per frame.
pub(crate) struct OutgoingBatch {
    tasks: Vec<RemoteMessage>,
    threshold: usize,
}

impl OutgoingBatch {
    /// Create a buffer that signals a flush once `threshold` tasks accumulate.
    pub(crate) fn new(threshold: usize) -> Self {
        Self {
            tasks: Vec::with_capacity(threshold),
            threshold,
        }
    }

    /// Append a task. Returns `true` once the buffer has reached its flush threshold, i.e.
    /// the caller should [`take`](Self::take) and send.
    pub(crate) fn push(&mut self, task: RemoteMessage) -> bool {
        self.tasks.push(task);
        self.tasks.len() >= self.threshold
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Drain the accumulated tasks, leaving the buffer empty for the next batch.
    pub(crate) fn take(&mut self) -> Vec<RemoteMessage> {
        std::mem::take(&mut self.tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::Task;

    fn task() -> RemoteMessage {
        RemoteMessage::Task(Task::Seed(0))
    }

    #[test]
    fn push_signals_flush_only_at_threshold() {
        let mut batch = OutgoingBatch::new(3);
        assert!(!batch.push(task())); // 1
        assert!(!batch.push(task())); // 2
        assert!(batch.push(task())); // 3 == threshold
    }

    #[test]
    fn threshold_of_one_flushes_every_push() {
        let mut batch = OutgoingBatch::new(1);
        assert!(batch.push(task()));
        assert!(batch.push(task()));
    }

    #[test]
    fn take_drains_and_resets() {
        let mut batch = OutgoingBatch::new(4);
        assert!(batch.is_empty());

        batch.push(task());
        batch.push(task());
        assert!(!batch.is_empty());

        let drained = batch.take();
        assert_eq!(drained.len(), 2);

        // Taking resets the buffer, so a subsequent take yields nothing.
        assert!(batch.is_empty());
        assert_eq!(batch.take().len(), 0);
    }
}
