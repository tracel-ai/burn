use burn_std::id::StreamId;
use burn_communication::{Protocol, data_service::TensorDataService};
use burn_ir::BackendIr;
use burn_router::Runner;
use burn_tensor::Device;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};

use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponse};

use super::stream::Stream;

/// A session manager control the creation of sessions.
///
/// Each session manages its own stream, spawning one thread per stream to mimic the same behavior
/// a native backend would have.
pub struct SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    runner: Runner<B>,
    sessions: Mutex<HashMap<SessionId, Session<B, P>>>,
    data_service: Arc<TensorDataService<B, P>>,
}

struct Session<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    runner: Runner<B>,
    streams: HashMap<StreamId, Stream<B, P>>,
    sender: Sender<Receiver<TaskResponse>>,
    receiver: Option<Receiver<Receiver<TaskResponse>>>,
    data_service: Arc<TensorDataService<B, P>>,
}

impl<B, P> SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub fn new(device: Device<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        Self {
            runner: Runner::new(device),
            sessions: Mutex::new(Default::default()),
            data_service,
        }
    }

    /// Register a new responder for the session. Only one responder can exist for a session for
    /// now.
    pub async fn register_responder(
        &self,
        session_id: SessionId,
    ) -> Receiver<Receiver<TaskResponse>> {
        log::info!("Register responder for session {session_id}");
        let mut sessions = self.sessions.lock().await;
        self.register_session(&mut sessions, session_id);

        let session = sessions.get_mut(&session_id).unwrap();
        session.init_responder()
    }

    /// Get the stream for the current session and task.
    pub async fn stream(
        &self,
        session_id: &mut Option<SessionId>,
        task: Task,
    ) -> Option<(Stream<B, P>, ConnectionId, ComputeTask)> {
        let mut sessions = self.sessions.lock().await;

        let session_id = match session_id {
            Some(id) => *id,
            None => match task {
                Task::Init(id) => {
                    log::info!("Init requester for session {id}");
                    *session_id = Some(id);
                    self.register_session(&mut sessions, id);
                    return None;
                }
                _ => panic!("The first message should initialize the session"),
            },
        };

        match sessions.get_mut(&session_id) {
            Some(session) => {
                let (task, connection_id) = match task {
                    Task::Compute(task, connection_id) => (task, connection_id),
                    _ => panic!("Only support compute tasks."),
                };
                let stream = session.select(connection_id.stream_id).await;
                Some((stream, connection_id, task))
            }
            None => panic!("To be initialized"),
        }
    }

    /// Close the session with the given id.
    pub async fn close(&self, session_id: Option<SessionId>) {
        if let Some(id) = session_id {
            let mut sessions = self.sessions.lock().await;
            if let Some(session) = sessions.get_mut(&id) {
                session.close().await;
            }
        }
    }

    fn register_session(&self, sessions: &mut HashMap<SessionId, Session<B, P>>, id: SessionId) {
        sessions.entry(id).or_insert_with(|| {
            log::info!("Creating a new session {id}");

            Session::new(self.runner.clone(), self.data_service.clone())
        });
    }
}

impl<B, P> Session<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    fn new(runner: Runner<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);

        Self {
            runner,
            streams: Default::default(),
            sender,
            receiver: Some(receiver),
            data_service,
        }
    }

    fn init_responder(&mut self) -> Receiver<Receiver<TaskResponse>> {
        let mut receiver = None;
        core::mem::swap(&mut receiver, &mut self.receiver);
        receiver.expect("Only one responder per session is possible.")
    }

    /// Select the current [stream](Stream) based on the given task.
    async fn select(&mut self, stream_id: StreamId) -> Stream<B, P> {
        // We return the stream.
        match self.streams.get(&stream_id) {
            Some(stream) => stream.clone(),
            None => {
                let stream = Stream::<B, P>::new(
                    self.runner.clone(),
                    self.sender.clone(),
                    self.data_service.clone(),
                )
                .await;
                self.streams.insert(stream_id, stream.clone());
                stream
            }
        }
    }

    // Close all streams created in the session.
    async fn close(&mut self) {
        for (id, stream) in self.streams.drain() {
            log::info!("Closing stream {id}");
            stream.close().await;
        }
    }
}
