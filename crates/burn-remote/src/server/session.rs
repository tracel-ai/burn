use burn_backend::tensor::Device;
use burn_communication::{Protocol, data_service::TensorDataService};
use burn_ir::BackendIr;
use burn_router::TensorInterpreter;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, mpsc};

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
    device: Device<B>,
    pub(crate) data_service: Arc<TensorDataService<B, P>>,
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
    pub fn new(device: Device<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        Self {
            device,
            data_service,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// The backend default device settings, used by the response-init handshake before any
    /// session-specific runner is needed.
    pub fn device_settings(&self) -> burn_std::DeviceSettings {
        use burn_backend::backend::DeviceOps;
        self.device.defaults()
    }

    /// Resolve both the [`TensorInterpreter`] and the response sender for `session_id` in a single lock
    /// acquisition, creating the session on demand. The request loop resolves these once per
    /// connection and reuses them for every task, instead of re-locking the sessions map (and
    /// re-cloning the runner) per task.
    pub async fn session_handles(
        &self,
        session_id: SessionId,
    ) -> (TensorInterpreter<B>, mpsc::Sender<TaskResponse>) {
        self.with_session(session_id, |s| (s.runner.clone(), s.sender.clone()))
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
    ) -> Result<mpsc::Receiver<TaskResponse>, String> {
        self.with_session(session_id, |s| {
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
        f: impl FnOnce(&mut Session<B>) -> R,
    ) -> R {
        let mut sessions = self.sessions.lock().await;
        let entry = sessions.entry(session_id).or_insert_with(|| {
            let (sender, receiver) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            Session {
                runner: TensorInterpreter::new(self.device.clone()),
                sender,
                receiver: Some(receiver),
            }
        });
        f(entry)
    }
}
