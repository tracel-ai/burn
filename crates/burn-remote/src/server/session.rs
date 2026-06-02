use burn_backend::tensor::Device;
use burn_communication::{Protocol, data_service::TensorDataService};
use burn_ir::BackendIr;
use burn_router::TensorInterpreter;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, mpsc};

use crate::server::local_transfer::LocalTransferService;
use crate::shared::{SessionId, TaskResponse};

/// Capacity for the per-session response queue.
///
/// Sized larger than the typical in-flight read/sync count so that request processing
/// doesn't block on backpressure during a burst, but small enough that a stuck response
/// writer surfaces as a backpressure stall rather than memory growth.
const RESPONSE_CHANNEL_CAPACITY: usize = 64;

/// Coordinates per-session state.
///
/// Each [`Session`] owns its own [`TensorInterpreter`] with its own [`HandleContainer`](burn_ir::HandleContainer)
/// — different sessions never share tensor handles, so concurrent sessions can't race on
/// each other's backend state. Cross-session tensor transfers go through `data_service`,
/// which already serializes the bytes through its own protocol.
///
/// Within a session there is exactly one request-handling task (one tokio task per socket
/// connection), so per-session ordering is preserved without any extra locking.
pub struct SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// All devices this server hosts, indexed by the device index the client selects at
    /// session init. `devices[0]` is the default device (`DeviceIndex::Default`).
    devices: Vec<Device<B>>,
    pub(crate) data_service: Arc<TensorDataService<B, P>>,
    /// Rendezvous registry for same-host tensor transfers between this server's sessions.
    pub(crate) local_transfers: Arc<LocalTransferService<B>>,
    sessions: Mutex<HashMap<SessionId, Session<B>>>,
}

struct Session<B: BackendIr> {
    runner: TensorInterpreter<B>,
    sender: mpsc::Sender<TaskResponse>,
    receiver: Option<mpsc::Receiver<TaskResponse>>,
}

impl<B, P> SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub fn new(devices: Vec<Device<B>>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        assert!(
            !devices.is_empty(),
            "A remote server must host at least one device"
        );
        Self {
            devices,
            data_service,
            local_transfers: Arc::new(LocalTransferService::new()),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve the device at `device_index`, falling back to the default device (index 0)
    /// with a warning if the client requested an index this server doesn't host.
    fn device(&self, device_index: u32) -> Device<B> {
        match self.devices.get(device_index as usize) {
            Some(device) => device.clone(),
            None => {
                log::warn!(
                    "Requested device index {device_index} but server hosts only {} device(s); \
                     falling back to device 0",
                    self.devices.len()
                );
                self.devices[0].clone()
            }
        }
    }

    /// The device settings for `device_index`, used by the response-init handshake before any
    /// session-specific runner is needed.
    pub fn device_settings(&self, device_index: u32) -> burn_std::DeviceSettings {
        use burn_backend::backend::DeviceOps;
        self.device(device_index).defaults()
    }

    /// Resolve both the [`TensorInterpreter`] and the response sender for `session_id` in a single lock
    /// acquisition, creating the session on demand. The request loop resolves these once per
    /// connection and reuses them for every task, instead of re-locking the sessions map (and
    /// re-cloning the runner) per task.
    pub async fn session_handles(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> (TensorInterpreter<B>, mpsc::Sender<TaskResponse>) {
        self.with_session(session_id, device_index, |s| {
            (s.runner.clone(), s.sender.clone())
        })
        .await
    }

    /// Get a clone of the [`TensorInterpreter`] for `session_id` only if the session already exists,
    /// without creating one. Used on the close path so a `Close` for an unknown or
    /// already-removed session doesn't resurrect a phantom runner just to drop it.
    pub async fn try_runner(&self, session_id: SessionId) -> Option<TensorInterpreter<B>> {
        let sessions = self.sessions.lock().await;
        sessions.get(&session_id).map(|s| s.runner.clone())
    }

    /// Take the response receiver for `session_id`.
    ///
    /// Returns `Err` if a responder has already been registered for this session — the
    /// protocol allows only one response socket per session.
    pub async fn take_response_receiver(
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

    /// Drop the session. The runner's handles are released here, so any backend state held
    /// only by this session's tensors becomes eligible for cleanup.
    pub async fn close(&self, session_id: SessionId) {
        let mut sessions = self.sessions.lock().await;
        sessions.remove(&session_id);
    }

    async fn with_session<R>(
        &self,
        session_id: SessionId,
        device_index: u32,
        f: impl FnOnce(&mut Session<B>) -> R,
    ) -> R {
        let mut sessions = self.sessions.lock().await;
        let entry = sessions.entry(session_id).or_insert_with(|| {
            let (sender, receiver) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            Session {
                // The session is pinned to its device for its whole lifetime; the first
                // handler (request or response) to touch it fixes the device index.
                runner: TensorInterpreter::new(self.device(device_index)),
                sender,
                receiver: Some(receiver),
            }
        });
        f(entry)
    }
}
