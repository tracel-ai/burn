//! The `/submit` connection handler.

use std::future::Future;
use std::sync::Arc;

use burn_communication::CommunicationChannel;
use tokio::sync::mpsc;

use super::policy::SubmitPolicy;
use crate::shared::{RemoteMessage, SessionId, Task};

/// What a `/submit` connection needs from the session layer: the worker channel for a session,
/// and a way to tear a session down. Async methods return `impl Future + Send` (as the
/// [`CommunicationChannel`] trait does) so a handler future built on them stays `Send` and can
/// be spawned by the server.
pub(crate) trait SubmitService: Send + Sync + 'static {
    /// The channel forwarding tasks to `session_id`'s worker, creating the session (and
    /// spawning its worker) on demand.
    fn session_task_sender(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = mpsc::Sender<Task>> + Send;

    /// Drop the session, letting its worker drain and exit. A `close` for an unknown session is
    /// a no-op.
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
