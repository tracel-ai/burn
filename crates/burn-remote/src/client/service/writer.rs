//! Outgoing-frame writer task.

use burn_communication::{CommunicationChannel, Message, ProtocolClient};
use tokio::sync::mpsc;

/// Bound on serialized request frames queued for the writer task.
///
/// [`send`](RequestWriter::send) enqueues here instead of touching the socket directly.
/// Bounded so a stalled socket surfaces as backpressure on the runner thread (a `send`
/// blocks only when the writer has fallen this many frames behind) rather than as unbounded
/// memory growth.
const WRITE_QUEUE_CAP: usize = 16;

/// Owns the request channel on the service runtime and serializes outgoing frames.
///
/// The runner thread hands serialized batches to [`send`](Self::send) over a bounded
/// channel; the writer task `await`s each socket send fully before pulling the next, so
/// frames reach the wire in FIFO order without ever parking the runner thread on the
/// network. That single-task FIFO drain is also what guarantees a frame is fully flushed
/// before the next begins — the websocket sink itself offers no such queue.
pub(crate) struct RequestWriter {
    /// `Option` so [`shutdown`](Self::shutdown) can drop the sender to signal the task to
    /// finish once it has drained.
    tx: Option<mpsc::Sender<bytes::Bytes>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl RequestWriter {
    /// Spawn the writer task on `runtime`, taking ownership of the request `channel`.
    pub(crate) fn spawn<C: ProtocolClient>(
        runtime: &tokio::runtime::Runtime,
        mut channel: C::Channel,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<bytes::Bytes>(WRITE_QUEUE_CAP);
        let handle = runtime.spawn(async move {
            while let Some(bytes) = rx.recv().await {
                if let Err(err) = channel.send(Message::new(bytes)).await {
                    log::warn!("Remote request writer send failed: {err:?}; closing writer");
                    return;
                }
            }
        });
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Enqueue a serialized frame for the wire. Blocks the calling (runner) thread only when
    /// the writer is [`WRITE_QUEUE_CAP`] frames behind (socket backpressure), never on a
    /// healthy send.
    pub(crate) fn send(&self, runtime: &tokio::runtime::Runtime, frame: bytes::Bytes) {
        let Some(tx) = self.tx.as_ref() else {
            log::warn!("Remote request writer already shut down; dropping outgoing frame");
            return;
        };
        // The writer task exits if a socket send fails (server gone / connection reset),
        // after which `tx.send` errors. Drop the frame with a warning rather than panicking
        // here — `send` runs on the device-runner thread that executes user ops, so a panic
        // would crash the caller's thread instead of surfacing a recoverable disconnect.
        if runtime.block_on(tx.send(frame)).is_err() {
            log::warn!(
                "Remote request writer task has exited (server disconnected?); dropping outgoing frame"
            );
        }
    }

    /// Best-effort teardown: enqueue an optional final frame, drop the sender so the writer
    /// drains and exits, then join it so the runtime isn't torn down mid-send.
    pub(crate) fn shutdown(
        &mut self,
        runtime: &tokio::runtime::Runtime,
        final_frame: Option<bytes::Bytes>,
    ) {
        if let (Some(frame), Some(tx)) = (final_frame, self.tx.as_ref()) {
            let _ = runtime.block_on(tx.send(frame));
        }
        // Drop the sender so the writer's `rx.recv()` returns `None` once it has drained.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = runtime.block_on(handle);
        }
    }
}
