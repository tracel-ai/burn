use burn_backend::tensor::Device;
use burn_communication::{Protocol, data_service::TensorDataService};
use burn_ir::BackendIr;
use burn_router::Runner;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, mpsc};

use crate::shared::{SessionId, TaskResponse};

/// Capacity for the per-session response queue.
///
/// Sized larger than the typical in-flight read/sync count so that request processing
/// doesn't block on backpressure during a burst, but small enough that a stuck response
/// writer surfaces as a backpressure stall rather than memory growth.
const RESPONSE_CHANNEL_CAPACITY: usize = 64;

/// Coordinates the per-session response channel.
///
/// All actual backend execution shares the single [`Runner`] owned by the manager — the
/// runner's internal mutex serializes ops across sessions/streams. Each session has one
/// `mpsc` pair: the **sender** is cloned per-task to publish responses (from the request
/// handler), and the **receiver** is taken once by the response-writing task.
///
/// There is no per-session thread, no per-stream processor, and no nested callback queue.
/// Ops are executed inline on the request-handling task with [`StreamId::executes`] used
/// to thread the client-side stream id through to the runner.
pub struct SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub(crate) runner: Runner<B>,
    pub(crate) data_service: Arc<TensorDataService<B, P>>,
    sessions: Mutex<HashMap<SessionId, SessionChannel>>,
}

struct SessionChannel {
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
            runner: Runner::new(device),
            data_service,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Get a clone of the response sender for `session_id`, creating the channel on demand
    /// if this is the first time we see this session.
    pub async fn response_sender(&self, session_id: SessionId) -> mpsc::Sender<TaskResponse> {
        self.with_session(session_id, |s| s.sender.clone()).await
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
            s.receiver.take().ok_or_else(|| {
                format!("Response receiver already taken for session {session_id}")
            })
        })
        .await
    }

    /// Forget a session. Subsequent attempts to send a response to its sender will fail
    /// once the receiver is dropped — that signal naturally tears the response writer
    /// down.
    pub async fn close(&self, session_id: SessionId) {
        let mut sessions = self.sessions.lock().await;
        sessions.remove(&session_id);
    }

    async fn with_session<R>(
        &self,
        session_id: SessionId,
        f: impl FnOnce(&mut SessionChannel) -> R,
    ) -> R {
        let mut sessions = self.sessions.lock().await;
        let entry = sessions.entry(session_id).or_insert_with(|| {
            let (sender, receiver) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            SessionChannel {
                sender,
                receiver: Some(receiver),
            }
        });
        f(entry)
    }
}
