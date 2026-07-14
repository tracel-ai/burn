//! Outgoing task buffering.

use crate::shared::{RemoteMessage, Task};

/// Accumulates outgoing [`RemoteMessage`]s on the runner thread and decides when a batch is ready
/// for the wire.
///
/// Pure buffering logic — no runtime, no socket. [`push`](Self::push) reports when a flush
/// threshold is reached so the caller can flush; [`take`](Self::take) drains the buffer into a
/// frame, leaving it empty for the next batch.
///
/// Two thresholds drive a flush, whichever hits first:
/// - **task count** — bounds latency for streams of many tiny ops;
/// - **buffered data bytes** — bounds how much tensor data sits unsent, so a big upload (or several
///   moderate ones) goes out promptly instead of waiting on the count threshold. Tracking a running
///   sum, rather than flushing on each individual large message, lets small tensors keep batching
///   while still capping the in-flight data.
///
/// Batching is wire-level only: every task keeps its own `StreamId`/`RequestId`, so the
/// server sees the same per-task semantics whether tasks arrive batched or one per frame.
pub(crate) struct OutgoingBatch {
    tasks: Vec<RemoteMessage>,
    threshold: usize,
    /// Running sum of [`data_len`] over the buffered tasks; reset by [`take`](Self::take).
    bytes: usize,
    bytes_threshold: usize,
}

/// Size of the bulk tensor data a message carries, in bytes (0 for metadata-only messages). Drives
/// the byte-based flush threshold.
fn data_len(msg: &RemoteMessage) -> usize {
    match msg {
        RemoteMessage::Task(Task::RegisterTensor(_, _, data)) => data.bytes.len(),
        _ => 0,
    }
}

impl OutgoingBatch {
    /// Create a buffer that signals a flush once `threshold` tasks *or* `bytes_threshold` buffered
    /// data bytes accumulate.
    pub(crate) fn new(threshold: usize, bytes_threshold: usize) -> Self {
        Self {
            tasks: Vec::with_capacity(threshold),
            threshold,
            bytes: 0,
            bytes_threshold,
        }
    }

    /// Append a task. Returns `true` once the buffer has reached either flush threshold, i.e.
    /// the caller should [`take`](Self::take) and send.
    pub(crate) fn push(&mut self, task: RemoteMessage) -> bool {
        self.bytes += data_len(&task);
        self.tasks.push(task);
        self.tasks.len() >= self.threshold || self.bytes >= self.bytes_threshold
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Drain the accumulated tasks, leaving the buffer empty for the next batch.
    pub(crate) fn take(&mut self) -> Vec<RemoteMessage> {
        self.bytes = 0;
        std::mem::take(&mut self.tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::Task;
    use burn_backend::{StreamId, TensorData};
    use burn_ir::TensorId;

    /// A metadata-only task (contributes 0 to the byte counter).
    fn task() -> RemoteMessage {
        RemoteMessage::Task(Task::Seed(0))
    }

    /// A `RegisterTensor` carrying `n` bytes of data.
    fn data_task(n: usize) -> RemoteMessage {
        RemoteMessage::Task(Task::RegisterTensor(
            StreamId::current(),
            TensorId::new(0),
            TensorData::new(vec![0u8; n], [n]),
        ))
    }

    // A large byte threshold so these count-driven tests aren't affected by the byte trigger.
    const NO_BYTES: usize = usize::MAX;

    #[test]
    fn push_signals_flush_only_at_threshold() {
        let mut batch = OutgoingBatch::new(3, NO_BYTES);
        assert!(!batch.push(task())); // 1
        assert!(!batch.push(task())); // 2
        assert!(batch.push(task())); // 3 == threshold
    }

    #[test]
    fn threshold_of_one_flushes_every_push() {
        let mut batch = OutgoingBatch::new(1, NO_BYTES);
        assert!(batch.push(task()));
        assert!(batch.push(task()));
    }

    #[test]
    fn flushes_once_buffered_bytes_reach_threshold() {
        // Count threshold high enough that only the byte threshold can trigger.
        let mut batch = OutgoingBatch::new(100, 1000);
        assert!(!batch.push(data_task(400))); // 400 buffered
        assert!(!batch.push(data_task(400))); // 800 buffered
        assert!(batch.push(data_task(400))); // 1200 >= 1000 -> flush
    }

    #[test]
    fn take_resets_the_byte_counter() {
        let mut batch = OutgoingBatch::new(100, 1000);
        assert!(!batch.push(data_task(900)));
        batch.take(); // drains tasks and resets the byte sum
        // Counter restarted from zero, so 900 again does not trip the 1000-byte threshold.
        assert!(!batch.push(data_task(900)));
    }

    #[test]
    fn take_drains_and_resets() {
        let mut batch = OutgoingBatch::new(4, NO_BYTES);
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
