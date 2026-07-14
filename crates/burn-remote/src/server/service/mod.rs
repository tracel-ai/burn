//! The session-layer service trait, plus the init-handshake helper shared by the session pump.
//!
//! The pump ([`drive_session`](crate::server::pump::drive_session)) is written against
//! [`SessionService`] rather than a concrete type, so the submit-loop teardown logic stays testable
//! against a fake service with no backend and no live socket.

mod base;

pub(crate) use base::parse_init_handshake;

use std::future::Future;

use burn_std::DeviceSettings;
use tokio::sync::mpsc;

use crate::shared::{SessionId, Task, TaskResponse};

/// What the session pump needs from the session layer: bind a session to its worker channel, claim
/// its response receiver, read the device metadata for the handshake, and tear a session down.
///
/// Async methods return `impl Future + Send` so a session future built on them stays `Send` and can
/// be spawned by the server. The production implementation is
/// [`SessionManager`](super::session::SessionManager).
pub(crate) trait SessionService: Send + Sync + 'static {
    /// The channel forwarding tasks to `session_id`'s worker, creating the session (and spawning
    /// its worker) on demand.
    fn session_task_sender(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = mpsc::Sender<Task>> + Send;

    /// Claim the session's result receiver. Errors if a receiver was already taken — the protocol
    /// allows only one session stream per session.
    fn take_response_receiver(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = Result<mpsc::Receiver<TaskResponse>, String>> + Send;

    /// The default settings of the device at `device_index`, returned on the handshake so the
    /// client can resolve op dtypes without an extra round-trip.
    fn device_settings(&self, device_index: u32) -> DeviceSettings;

    /// The number of devices this server hosts, returned on the handshake so the client can
    /// enumerate every device behind the address.
    fn device_count(&self) -> u32;

    /// Drop the session, letting its worker drain and exit. A `close` for an unknown session is a
    /// no-op.
    fn close(&self, session_id: SessionId) -> impl Future<Output = ()> + Send;
}
