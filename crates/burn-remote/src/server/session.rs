use burn_common::id::StreamId;
use burn_router::Runner;
use burn_tensor::{
    ops::FullPrecisionBackend,
    repr::{ReprBackend, TensorDescription, TensorId, TensorStatus},
    Device,
};
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, Sender},
};
use tokio::sync::Mutex;

use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponse};

use super::stream::Stream;

/// A session manager control the creation of sessions.
///
/// Each session manages its own stream, spawning one thread per stream to mimic the same behavior
/// a native backend would have.
pub struct SessionManager<B: ReprBackend> {
    runner: Runner<B>,
    sessions: tokio::sync::Mutex<HashMap<SessionId, Session<B>>>,
}

struct Session<B: ReprBackend> {
    runner: Runner<B>,
    tensors: HashMap<TensorId, Vec<StreamId>>,
    streams: HashMap<StreamId, Stream<B>>,
    sender: Sender<Receiver<TaskResponse>>,
    receiver: Option<Receiver<Receiver<TaskResponse>>>,
}

impl<B: ReprBackend> SessionManager<B>
where
    // Restrict full precision backend handle to be the same
    FullPrecisionBackend<B>: ReprBackend<Handle = B::Handle>,
{
    pub fn new(device: Device<B>) -> Self {
        Self {
            runner: Runner::new(device),
            sessions: Mutex::new(Default::default()),
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
    ) -> Option<(Stream<B>, ConnectionId, ComputeTask)> {
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
                let stream = session.select(connection_id.stream_id, &task);
                Some((stream, connection_id, task))
            }
            None => {
                panic!("To be initialized");
            }
        }
    }

    /// Close the session with the given id.
    pub async fn close(&self, session_id: Option<SessionId>) {
        if let Some(id) = session_id {
            let mut sessions = self.sessions.lock().await;
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

impl<B: ReprBackend> Session<B>
where
    // Restrict full precision backend handle to be the same
    FullPrecisionBackend<B>: ReprBackend<Handle = B::Handle>,
{
    fn new(runner: Runner<B>) -> Self {
        let (sender, reveiver) = std::sync::mpsc::channel();
        Self {
            runner,
            tensors: Default::default(),
            streams: Default::default(),
            sender,
            receiver: Some(reveiver),
        }
    }

    fn init_responder(&mut self) -> Receiver<Receiver<TaskResponse>> {
        let mut receiver = None;
        core::mem::swap(&mut receiver, &mut self.receiver);
        receiver.expect("Only one responder per session is possible.")
    }

    /// Select the current [stream](Stream) based on the given task.
    fn select(&mut self, stream_id: StreamId, task: &ComputeTask) -> Stream<B> {
        // We have to check every streams involved in the last operation, making
        // sure the backend is up-to-date with those operations.
        //
        // 1. We update the tensor status of all tensors in the task.
        // 2. We don't keep track of tensors that are used for the last time.
        let mut fences = Vec::new();
        for (tensor_id, status) in task.tensors_info() {
            let tensor_stream_ids = match self.tensors.get(&tensor_id) {
                Some(val) => val,
                None => {
                    if status != TensorStatus::ReadWrite {
                        // Add the first stream that created the tensor that may be used by other
                        // streams later.
                        self.register_tensor(tensor_id, stream_id);
                    }
                    continue;
                }
            };

            let current_stream_already_synced = tensor_stream_ids.contains(&stream_id);

            if !current_stream_already_synced {
                // We only need to sync to the first stream that created the tensor.
                if let Some(id) = tensor_stream_ids.iter().next() {
                    fences.push(*id);
                }
            }

            // We add the stream to the list of updated stream to avoid needed to flush other
            // operations that might use this tensor.
            self.register_tensor(tensor_id, stream_id);

            // If the tensor has the status `read_write`, it means no other stream can reuse it
            // afterward, so we remove it from the state.
            if status == TensorStatus::ReadWrite {
                self.tensors.remove(&tensor_id);
            }
        }

        // Cleanup orphans.
        if let ComputeTask::RegisterOrphan(tensor_id) = task {
            self.tensors.remove(tensor_id);
        }

        // We have to wait for the streams to be updated.
        for stream_id in fences {
            if let Some(stream) = self.streams.get(&stream_id) {
                stream.fence_sync();
            }
        }

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

    fn register_tensor(&mut self, tensor_id: TensorId, stream_id: StreamId) {
        match self.tensors.get_mut(&tensor_id) {
            Some(ids) => {
                ids.push(stream_id);
            }
            None => {
                self.tensors.insert(tensor_id, vec![stream_id]);
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

impl ComputeTask {
    fn tensors_info(&self) -> Vec<(TensorId, TensorStatus)> {
        fn from_descriptions(desc: &[&TensorDescription]) -> Vec<(TensorId, TensorStatus)> {
            desc.iter().map(|t| (t.id, t.status.clone())).collect()
        }

        match self {
            ComputeTask::RegisterOperation(op) => from_descriptions(&op.nodes()),
            ComputeTask::RegisterTensor(tensor_id, _tensor_data) => {
                vec![(*tensor_id, TensorStatus::NotInit)]
            }
            ComputeTask::RegisterOrphan(tensor_id) => {
                vec![(*tensor_id, TensorStatus::ReadWrite)]
            }
            ComputeTask::ReadTensor(tensor_description) => from_descriptions(&[tensor_description]),
            ComputeTask::SyncBackend => vec![],
        }
    }
}
