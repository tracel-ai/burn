use burn_backend::tensor::Device;
use burn_communication::{Protocol, external_comm::ExternalCommService};
use burn_ir::BackendIr;
use burn_router::TensorInterpreter;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, mpsc};

use crate::server::local_comm::LocalCommService;
use crate::server::service::{FetchService, SubmitService};
use crate::server::worker::SessionHandler;
use crate::shared::{SessionId, Task, TaskResponse};

/// Capacity for the per-session response queue.
///
/// Sized larger than the typical in-flight read/sync count so that request processing
/// doesn't block on backpressure during a burst, but small enough that a stuck response
/// writer surfaces as a backpressure stall rather than memory growth.
const RESPONSE_CHANNEL_CAPACITY: usize = 64;

/// Coordinates per-session state.
///
/// Each [`Session`] owns a dedicated [`SessionHandler`] that holds the session's
/// [`TensorInterpreter`] with its own [`HandleContainer`](burn_ir::HandleContainer) — different
/// sessions never share tensor handles, so concurrent sessions can't race on each other's backend
/// state. Cross-session tensor transfers go through `external_comm` (cross-server) or `local_comm`
/// (same-host), each of which has its own rendezvous.
///
/// Tasks run on the handler's worker threads, not the submit handler: the latter only decodes the
/// incoming batch and forwards each [`Task`] to the session over a bounded channel. Inside the
/// session a dispatcher routes each task to a per-stream worker thread, so per-stream ordering is
/// preserved while independent streams — and other sessions — keep making progress even when one
/// stream is parked on a blocking op (a same-host transfer rendezvous or an all-reduce barrier).
pub struct SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// All devices this server hosts, indexed by the device index the client selects at
    /// session init. `devices[0]` is the default device (`DeviceIndex::Default`).
    devices: Vec<Device<B>>,
    pub(crate) external_comm: Arc<ExternalCommService<B, P>>,
    /// Rendezvous registry for same-host tensor transfers between this server's sessions.
    pub(crate) local_comm: Arc<LocalCommService<B>>,
    sessions: Mutex<HashMap<SessionId, Session>>,
}

struct Session {
    /// Inbound channel to the session's dispatcher thread; cloned once per submit connection.
    task_sender: mpsc::Sender<Task>,
    receiver: Option<mpsc::Receiver<TaskResponse>>,
}

impl<B, P> SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub fn new(devices: Vec<Device<B>>, external_comm: Arc<ExternalCommService<B, P>>) -> Self {
        assert!(
            !devices.is_empty(),
            "A remote server must host at least one device"
        );
        Self {
            devices,
            external_comm,
            local_comm: Arc::new(LocalCommService::new()),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve the device at `device_index`.
    ///
    /// The index is validated against the server's device count on the client init handshake, so
    /// an out-of-range index here is a protocol/configuration error (e.g. a client enumerating
    /// more devices than this server hosts). Fail loudly rather than silently collapsing onto
    /// device 0 — for a collective that would reduce a device against itself and silently corrupt
    /// the result instead of producing a clear failure.
    pub(crate) fn device(&self, device_index: u32) -> Device<B> {
        self.devices
            .get(device_index as usize)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "Requested device index {device_index} but server hosts only {} device(s)",
                    self.devices.len()
                )
            })
    }

    async fn with_session<R>(
        &self,
        session_id: SessionId,
        device_index: u32,
        f: impl FnOnce(&mut Session) -> R,
    ) -> R {
        let mut sessions = self.sessions.lock().await;
        let entry = sessions.entry(session_id).or_insert_with(|| {
            let (sender, receiver) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            // The session is pinned to its device for its whole lifetime; the first handler
            // (request or response) to touch it fixes the device index. Spawn the handler that
            // owns the runner — this runs inside the tokio runtime, so the handler can capture
            // the runtime handle its worker threads need for the async parts of a task.
            let runner = TensorInterpreter::new(self.device(device_index));
            let task_sender = SessionHandler::spawn(
                session_id,
                runner,
                sender,
                self.external_comm.clone(),
                self.local_comm.clone(),
            );
            Session {
                task_sender,
                receiver: Some(receiver),
            }
        });
        f(entry)
    }
}

impl<B, P> SubmitService for SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Resolve the channel used to forward [`Task`]s to `session_id`'s dispatcher thread,
    /// creating the session (and spawning its handler) on demand. The request loop resolves
    /// this once per connection and reuses it for every task, instead of re-locking the
    /// sessions map per task.
    async fn session_task_sender(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> mpsc::Sender<Task> {
        self.with_session(session_id, device_index, |s| s.task_sender.clone())
            .await
    }

    /// Drop the session, detaching its handler. Removing the map entry drops the inbound task
    /// sender it holds; once the request connection also drops its cloned task sender, the
    /// dispatcher's channel closes, so the per-stream workers drain and exit and the handler
    /// flushes its runner, releasing any backend state held only by this session's tensors.
    async fn close(&self, session_id: SessionId) {
        let mut sessions = self.sessions.lock().await;
        sessions.remove(&session_id);
    }
}

impl<B, P> FetchService for SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// The device settings for `device_index`, used by the response-init handshake before any
    /// session-specific runner is needed.
    fn device_settings(&self, device_index: u32) -> burn_std::DeviceSettings {
        use burn_backend::backend::DeviceOps;
        self.device(device_index).defaults()
    }

    /// The total number of devices this server hosts. Sent to the client on the init handshake
    /// so it can enumerate every device behind the address (see [`RemoteDevice::enumerate`]).
    fn device_count(&self) -> u32 {
        self.devices.len() as u32
    }

    /// Take the response receiver for `session_id`.
    ///
    /// Returns `Err` if a responder has already been registered for this session — the
    /// protocol allows only one response socket per session.
    async fn take_response_receiver(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> Result<mpsc::Receiver<TaskResponse>, String> {
        self.with_session(session_id, device_index, |s| {
            s.receiver
                .take()
                .ok_or_else(|| format!("Response receiver already taken for session {session_id}"))
        })
        .await
    }
}
