//! The `/submit` connection handler.

use std::future::Future;
use std::sync::Arc;

use burn_communication::CommunicationChannel;

use super::policy::SubmitPolicy;
use crate::shared::{RemoteMessage, SessionId, Task};

/// A sink that forwards a decoded [`Task`] to its session's device runner.
///
/// Returned by [`SubmitService::session_forwarder`]; calling it enqueues the task on the device's
/// cubecl runner channel — a synchronous, lock-free hand-off, so forwarding never `await`s.
pub(crate) type TaskForwarder = Box<dyn Fn(Task) + Send + Sync>;

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
