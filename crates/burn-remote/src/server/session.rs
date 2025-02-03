use burn_common::id::StreamId;
use burn_common::stub::Mutex;
use burn_router::Runner;
use burn_tensor::{repr::ReprBackend, Device};
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, SyncSender},
};

use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponse};

use super::stream::Stream;

/// A session manager control the creation of sessions.
///
/// Each session manages its own stream, spawning one thread per stream to mimic the same behavior
/// a native backend would have.
pub struct SessionManager<B: ReprBackend> {
    runner: Runner<B>,
    sessions: Mutex<HashMap<SessionId, Session<B>>>,
}

struct Session<B: ReprBackend> {
    runner: Runner<B>,
    streams: HashMap<StreamId, Stream<B>>,
    sender: SyncSender<Receiver<TaskResponse>>,
    receiver: Option<Receiver<Receiver<TaskResponse>>>,
}

impl<B: ReprBackend> SessionManager<B> {
    pub fn new(device: Device<B>) -> Self {
        Self {
            runner: Runner::new(device),
            sessions: Mutex::new(Default::default()),
        }
    }

    /// Register a new responder for the session. Only one responder can exist for a session for
    /// now.
    pub fn register_responder(&self, session_id: SessionId) -> Receiver<Receiver<TaskResponse>> {
        log::info!("Register responder for session {session_id}");
        let mut sessions = self.sessions.lock().unwrap();
        self.register_session(&mut sessions, session_id);

        let session = sessions.get_mut(&session_id).unwrap();
        session.init_responder()
    }

    /// Get the stream for the current session and task.
    pub fn stream(
        &self,
        session_id: &mut Option<SessionId>,
        task: Task,
    ) -> Option<(Stream<B>, ConnectionId, ComputeTask)> {
        let mut sessions = self.sessions.lock().unwrap();

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
                let stream = session.select(connection_id.stream_id);
                Some((stream, connection_id, task))
            }
            None => panic!("To be initialized"),
        }
    }

    /// Close the session with the given id.
    pub fn close(&self, session_id: Option<SessionId>) {
        if let Some(id) = session_id {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(session) = sessions.get_mut(&id) {
                session.close();
            }
        }
    }

    fn register_session(&self, sessions: &mut HashMap<SessionId, Session<B>>, id: SessionId) {
        sessions.entry(id).or_insert_with(|| {
            log::info!("Creating a new session {id}");

            Session::new(self.runner.clone())
        });
    }
}

impl<B: ReprBackend> Session<B> {
    fn new(runner: Runner<B>) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        Self {
            runner,
            streams: Default::default(),
            sender,
            receiver: Some(receiver),
        }
    }

    fn init_responder(&mut self) -> Receiver<Receiver<TaskResponse>> {
        let mut receiver = None;
        core::mem::swap(&mut receiver, &mut self.receiver);
        receiver.expect("Only one responder per session is possible.")
    }

    /// Select the current [stream](Stream) based on the given task.
    fn select(&mut self, stream_id: StreamId) -> Stream<B> {
        // We return the stream.
        match self.streams.get(&stream_id) {
            Some(stream) => stream.clone(),
            None => {
                let stream = Stream::<B>::new(self.runner.clone(), self.sender.clone());
                self.streams.insert(stream_id, stream.clone());
                stream
            }
        }
    }

    // Close all streams created in the session.
    fn close(&mut self) {
        for (id, stream) in self.streams.drain() {
            log::info!("Closing stream {id}");
            stream.close();
        }
    }
}
