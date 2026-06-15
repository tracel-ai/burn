//! Outgoing-frame writer task.

use crate::shared::RemoteMessage;
use burn_communication::{CommunicationChannel, Message, ProtocolClient};
use tokio::sync::mpsc;

/// Bound on task batches queued for the writer task.
///
/// [`send`](SubmitWriter::send) enqueues here instead of touching the socket directly.
/// Bounded so a stalled socket surfaces as backpressure on the runner thread (a `send`
/// blocks only when the writer has fallen this many batches behind) rather than as unbounded
/// memory growth.
const WRITE_QUEUE_CAP: usize = 16;

/// Owns the submit channel on the service runtime and turns task batches into wire frames.
///
/// The runner thread hands raw [`RemoteMessage`] batches to [`send`](Self::send) over a bounded
/// channel; the writer task serializes and `await`s each socket send fully before pulling
/// the next, so frames reach the wire in FIFO order without ever parking the runner thread on
/// serialization or the network. Serializing here (rather than on the runner thread) lets
/// encoding one frame — which for `RegisterTensor` carries full tensor payloads — overlap
/// with the runner registering the next op. That single-task FIFO drain is also what
/// guarantees a frame is fully flushed before the next begins — the websocket sink itself
/// offers no such queue.
pub(crate) struct SubmitWriter {
    /// `Option` so [`shutdown`](Self::shutdown) can drop the sender to signal the task to
    /// finish once it has drained.
    tx: Option<mpsc::Sender<Vec<RemoteMessage>>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl SubmitWriter {
    /// Spawn the writer task on `runtime`, taking ownership of the submit `channel`.
    pub(crate) fn spawn<C: ProtocolClient>(
        runtime: &tokio::runtime::Runtime,
        mut channel: C::Channel,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<Vec<RemoteMessage>>(WRITE_QUEUE_CAP);
        let handle = runtime.spawn(async move {
            while let Some(batch) = rx.recv().await {
                let bytes: bytes::Bytes = match rmp_serde::to_vec(&batch) {
                    Ok(b) => b.into(),
                    Err(err) => {
                        log::error!("Failed to serialize outgoing task batch: {err:?}; dropping");
                        continue;
                    }
                };
                if let Err(err) = channel.send(Message::new(bytes)).await {
                    log::warn!("Remote submit writer send failed: {err:?}; closing writer");
                    return;
                }
            }
        });
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Enqueue a task batch for serialization + the wire.
    ///
    /// Fast path is a non-blocking [`try_send`](mpsc::Sender::try_send) — no tokio runtime
    /// hop. Only when the writer has fallen [`WRITE_QUEUE_CAP`] batches behind (socket
    /// backpressure) do we block the runner thread waiting for room.
    pub(crate) fn send(&self, runtime: &tokio::runtime::Runtime, batch: Vec<RemoteMessage>) {
        let Some(tx) = self.tx.as_ref() else {
            log::warn!("Remote submit writer already shut down; dropping outgoing batch");
            return;
        };
        match tx.try_send(batch) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(batch)) => {
                // Backpressure: wait for the writer to drain a slot. FIFO order is preserved
                // because this runs on the single runner thread, after the fast-path sends.
                if runtime.block_on(tx.send(batch)).is_err() {
                    log::warn!(
                        "Remote submit writer task has exited (server disconnected?); dropping outgoing batch"
                    );
                }
            }
            // The writer task exits if a socket send fails (server gone / connection reset).
            // Drop the batch with a warning rather than panicking on the runner thread, which
            // would crash the caller's thread instead of surfacing a recoverable disconnect.
            Err(mpsc::error::TrySendError::Closed(_)) => {
                log::warn!(
                    "Remote submit writer task has exited (server disconnected?); dropping outgoing batch"
                );
            }
        }
    }

    /// Best-effort teardown: enqueue an optional final batch, drop the sender so the writer
    /// drains and exits, then join it so the runtime isn't torn down mid-send.
    pub(crate) fn shutdown(
        &mut self,
        runtime: &tokio::runtime::Runtime,
        final_batch: Option<Vec<RemoteMessage>>,
    ) {
        if let (Some(batch), Some(tx)) = (final_batch, self.tx.as_ref()) {
            let _ = runtime.block_on(tx.send(batch));
        }
        // Drop the sender so the writer's `rx.recv()` returns `None` once it has drained.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = runtime.block_on(handle);
        }
    }
}
