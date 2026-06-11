//! The `/submit` connection handler.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use burn_communication::CommunicationChannel;

use super::policy::SubmitPolicy;
use crate::shared::{RemoteMessage, SessionId, Task};

/// The future returned when forwarding a task that must be carried to completion.
pub(crate) type BlockingForward = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A sink that forwards decoded [`Task`]s to a session's device runner, mirroring the device
/// handle's two dispatch modes so the runner's batching isn't defeated.
///
/// Returned by [`SubmitService::session_forwarder`]. The device runner's channel *batches*
/// enqueued closures and only runs a batch once it is flushed (or fills); flushing on every task
/// would throw away that batching — the main reason the server rides the device handle — so the
/// two modes are split by who is waiting on the result:
///
/// - [`submit`](Self::submit): fire-and-forget ops (op registration, tensor data, seed). Enqueued
///   on the batched channel and left to ride along; nobody is blocked on them, and the next
///   [`submit_blocking`](Self::submit_blocking) (a read/sync) flushes the batch that ran them.
/// - [`submit_blocking`](Self::submit_blocking): a task someone is waiting on — a result-producing
///   task (read/sync/dtype), or a tensor transfer. It flushes the batch and runs to completion.
///   For transfers the rendezvous/download is awaited *off the runner* (so the runner never blocks
///   and can't deadlock under concurrent cross-device transfers) before the register is enqueued.
pub(crate) struct TaskForwarder {
    submit: Box<dyn Fn(Task) + Send + Sync>,
    submit_blocking: Box<dyn Fn(Task) -> BlockingForward + Send + Sync>,
}

impl TaskForwarder {
    pub(crate) fn new(
        submit: impl Fn(Task) + Send + Sync + 'static,
        submit_blocking: impl Fn(Task) -> BlockingForward + Send + Sync + 'static,
    ) -> Self {
        Self {
            submit: Box::new(submit),
            submit_blocking: Box::new(submit_blocking),
        }
    }

    /// Enqueue a fire-and-forget task on the runner's batched channel. Does not flush, so the task
    /// rides the current batch until a [`submit_blocking`](Self::submit_blocking) flushes it.
    pub(crate) fn submit(&self, task: Task) {
        (self.submit)(task);
    }

    /// Forward a task that must complete before the connection moves on, returning once it has:
    /// the batch is flushed and the task runs, with any transfer rendezvous awaited off the runner.
    pub(crate) async fn submit_blocking(&self, task: Task) {
        (self.submit_blocking)(task).await;
    }
}

/// Whether `task` is a synchronization point that must be carried to completion now
/// ([`submit_blocking`](TaskForwarder::submit_blocking)) rather than left to batch
/// ([`submit`](TaskForwarder::submit)).
///
/// Fire-and-forget compute (op registration, tensor data, seed) batches; everything a client or a
/// peer session is actively waiting on — reads, sync, dtype queries, and the two tensor-transfer
/// halves — is a sync point.
pub(crate) fn is_sync_point(task: &Task) -> bool {
    !matches!(
        task,
        Task::RegisterOperation(..) | Task::RegisterTensor(..) | Task::Seed(..)
    )
}

/// What a `/submit` connection needs from the session layer: a forwarder for a session's tasks,
/// and a way to tear a session down. Async methods return `impl Future + Send` (as the
/// [`CommunicationChannel`] trait does) so a handler future built on them stays `Send` and can
/// be spawned by the server.
pub(crate) trait SubmitService: Send + Sync + 'static {
    /// The forwarder routing tasks to `session_id`'s device runner, creating the session on
    /// demand. Resolved once per connection and reused for every task.
    fn session_forwarder(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = TaskForwarder> + Send;

    /// Drop the session, releasing its interpreter and result queue. A `close` for an unknown
    /// session is a no-op.
    fn close(&self, session_id: SessionId) -> impl Future<Output = ()> + Send;
}

/// The `/submit` connection: decode each incoming batch of [`RemoteMessage`]s and forward the
/// tasks to the bound session's worker (via [`SubmitPolicy`]), tearing the session down when the
/// stream ends.
pub(crate) struct SubmitHandler<S, C> {
    socket: C,
    policy: SubmitPolicy<S>,
}

impl<S: SubmitService, C: CommunicationChannel> SubmitHandler<S, C> {
    pub(crate) fn new(service: Arc<S>, socket: C) -> Self {
        Self {
            socket,
            policy: SubmitPolicy::new(service),
        }
    }

    pub(crate) async fn run(mut self) {
        log::debug!(
            "[Submit handler] On new connection: {:?}",
            std::thread::current().id()
        );

        loop {
            let msg = match self.socket.recv().await {
                Ok(Some(m)) => m,
                Ok(None) => {
                    log::debug!("Submit stream closed");
                    break;
                }
                Err(err) => {
                    log::warn!("Submit stream error: {err:?}; closing");
                    break;
                }
            };

            let messages: Vec<RemoteMessage> = match rmp_serde::from_slice(&msg.data) {
                Ok(m) => m,
                Err(err) => {
                    log::error!("Failed to decode submitted batch: {err:?}; closing stream");
                    break;
                }
            };

            if self.policy.process_batch(messages).await {
                break;
            }
        }

        self.policy.cleanup().await;
    }
}
