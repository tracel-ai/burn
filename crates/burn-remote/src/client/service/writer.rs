//! Outgoing-message writer task.

use crate::{
    client::{
        runtime::{Executor, SpawnHandle},
        service::SubmitChannel,
    },
    shared::{Encode, RemoteMessage},
    transport::message::MessageSink,
};
use tokio::sync::mpsc;

/// Bound on task batches queued for the writer task on native targets.
///
/// [`send`](SubmitWriter::send) enqueues here instead of touching the socket directly.
/// Bounded so a stalled socket surfaces as backpressure on the runner thread (a `send`
/// blocks only when the writer has fallen this many batches behind) rather than as unbounded
/// memory growth. The browser cannot block the runner, so the wasm build uses an unbounded
/// channel and never applies backpressure.
#[cfg(not(target_family = "wasm"))]
const WRITE_QUEUE_CAP: usize = 16;

#[cfg(not(target_family = "wasm"))]
type BatchSender = mpsc::Sender<Vec<RemoteMessage>>;
#[cfg(target_family = "wasm")]
type BatchSender = mpsc::UnboundedSender<Vec<RemoteMessage>>;

/// Owns the submit channel on the service runtime and turns task batches into messages.
///
/// The runner thread hands raw [`RemoteMessage`] batches to [`send`](Self::send) over a channel;
/// the writer task serializes and `await`s each socket send fully before pulling the next, so
/// messages reach the wire in FIFO order without ever parking the runner thread on serialization
/// or the network. Serializing here (rather than on the runner thread) lets encoding one message,
/// which for `RegisterTensor` carries full tensor payloads, overlap with the runner registering
/// the next op. That single-task FIFO drain is also what guarantees a message's frames are all
/// sent before the next message begins; the socket sink itself offers no such queue.
pub(crate) struct SubmitWriter {
    /// `Option` so [`shutdown`](Self::shutdown) can drop the sender to signal the task to
    /// finish once it has drained.
    tx: Option<BatchSender>,
    /// Joined on shutdown on native; on wasm the detached task drains on the event loop.
    handle: Option<SpawnHandle>,
}

impl SubmitWriter {
    /// Spawn the writer task on `runtime`, taking ownership of the submit `channel`, which must be
    /// past the handshake. A failed send runs `end_session`, so readers fail instead of waiting for
    /// replies to frames never sent.
    pub(crate) fn spawn(
        runtime: &Executor,
        channel: SubmitChannel,
        end_session: impl FnOnce() + Send + 'static,
    ) -> Self {
        #[cfg(not(target_family = "wasm"))]
        let (tx, mut rx) = mpsc::channel::<Vec<RemoteMessage>>(WRITE_QUEUE_CAP);
        #[cfg(target_family = "wasm")]
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<RemoteMessage>>();

        let mut channel = MessageSink::new(channel);
        let handle = runtime.spawn(async move {
            while let Some(batch) = rx.recv().await {
                let message = match batch.encode() {
                    Ok(message) => message,
                    Err(err) => {
                        log::error!("Failed to serialize outgoing task batch: {err:?}; dropping");
                        continue;
                    }
                };
                if let Err(err) = channel.send(message).await {
                    log::warn!("Remote submit writer send failed: {err:?}; closing writer");
                    end_session();
                    return;
                }
            }
            let _ = channel.close().await;
        });
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Enqueue a task batch for serialization + the wire.
    ///
    /// Native: a non-blocking [`try_send`](mpsc::Sender::try_send) fast path; only when the writer
    /// has fallen [`WRITE_QUEUE_CAP`] batches behind (socket backpressure) do we block the runner
    /// thread waiting for room. Wasm: an unbounded, always non-blocking send.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn send(&self, runtime: &Executor, batch: Vec<RemoteMessage>) {
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

    #[cfg(target_family = "wasm")]
    pub(crate) fn send(&self, _runtime: &Executor, batch: Vec<RemoteMessage>) {
        let Some(tx) = self.tx.as_ref() else {
            log::warn!("Remote submit writer already shut down; dropping outgoing batch");
            return;
        };
        if tx.send(batch).is_err() {
            log::warn!(
                "Remote submit writer task has exited (server disconnected?); dropping outgoing batch"
            );
        }
    }

    /// Best-effort teardown: enqueue an optional final batch, drop the sender so the writer
    /// drains and exits, then (on native) join it so the runtime isn't torn down mid-send. In the
    /// browser the task is detached and finishes draining on the event loop after the sender drops.
    pub(crate) fn shutdown(&mut self, runtime: &Executor, final_batch: Option<Vec<RemoteMessage>>) {
        if let (Some(batch), Some(tx)) = (final_batch, self.tx.as_ref()) {
            #[cfg(not(target_family = "wasm"))]
            let _ = runtime.block_on(tx.send(batch));
            #[cfg(target_family = "wasm")]
            let _ = tx.send(batch);
        }
        // Drop the sender so the writer's `rx.recv()` returns `None` once it has drained.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            #[cfg(not(target_family = "wasm"))]
            runtime.join(handle);
            #[cfg(target_family = "wasm")]
            let _ = handle;
        }
        #[cfg(target_family = "wasm")]
        let _ = runtime;
    }
}

#[cfg(all(test, feature = "iroh", not(target_family = "wasm")))]
mod tests {
    use super::*;
    use iroh::{Endpoint, RelayMode, endpoint::presets};

    const ALPN: &[u8] = b"burn/writer-test";

    async fn local_endpoint() -> Endpoint {
        Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Disabled)
            .clear_ip_transports()
            .bind_addr("127.0.0.1:0")
            .unwrap()
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_send_ends_the_session() {
        let (server, client) = (local_endpoint().await, local_endpoint().await);
        let server_addr = server.addr();
        let accepted = tokio::spawn(async move { server.accept().await.unwrap().await.unwrap() });
        let connection = client.connect(server_addr, ALPN).await.unwrap();
        let (submit, _response) = connection.open_bi().await.unwrap();
        let _peer = accepted.await.unwrap();
        connection.close(0u32.into(), b"");

        let (ended, session_end) = tokio::sync::oneshot::channel();
        let executor = Executor::Tokio(tokio::runtime::Handle::current());
        let writer = SubmitWriter::spawn(&executor, SubmitChannel::Iroh(submit), move || {
            let _ = ended.send(());
        });
        writer.send(&executor, Vec::new());

        tokio::time::timeout(core::time::Duration::from_secs(10), session_end)
            .await
            .expect("the failed send ended the session")
            .unwrap();
    }
}
