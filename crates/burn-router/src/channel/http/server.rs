use core::marker::PhantomData;
use std::{collections::BTreeMap, sync::Arc};

use crate::{Runner, RunnerClient};
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{OperationDescription, ReprBackend, TensorDescription},
    Device, TensorData,
};
use tokio::sync::Mutex;

use super::{
    ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor, RegisterTensorEmpty, SyncBackend,
};

struct HttpServer<B: ReprBackend> {
    runner: Runner<B>,
}

enum QueuedTask {
    RegisterOperation(RegisterOperation),
    RegisterOrphan(RegisterOrphan),
    Executed,
}

type Callback<M> = std::sync::mpsc::Sender<M>;

enum ProcessorTask {
    RegisterOperation(OperationDescription),
    RegisterTensor(RegisterTensor, Callback<TensorDescription>),
    RegisterTensorEmpty(RegisterTensorEmpty, Callback<TensorDescription>),
    ReadTensor(ReadTensor, Callback<TensorData>),
    Sync(Callback<()>),
    RegisterOrphan(RegisterOrphan),
}

#[derive(Clone)]
struct Stream<B: ReprBackend> {
    sender: std::sync::mpsc::Sender<ProcessorTask>,
    queue: Arc<Mutex<Queue>>,
    _p: PhantomData<B>,
}

struct Queue {
    current_index: u64,
    tasks: BTreeMap<u64, QueuedTask>,
}

struct Processor<B: ReprBackend> {
    p: PhantomData<B>,
}

impl<B: ReprBackend> Processor<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> std::sync::mpsc::Sender<ProcessorTask> {
        let (sender, rec) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            for item in rec.iter() {
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        runner.register(op);
                    }
                    ProcessorTask::RegisterOrphan(val) => {
                        runner.register_orphan(&val.id);
                    }
                    ProcessorTask::Sync(callback) => {
                        runner.sync();
                        callback.send(()).unwrap();
                    }
                    ProcessorTask::RegisterTensor(val, callback) => {
                        let val = runner.register_tensor_data_desc(val.data);
                        callback.send(val).unwrap();
                    }
                    ProcessorTask::RegisterTensorEmpty(val, callback) => {
                        let val = runner.register_empty_tensor_desc(val.shape, val.dtype);
                        callback.send(val).unwrap();
                    }
                    ProcessorTask::ReadTensor(val, callback) => {
                        let tensor = burn_common::future::block_on(runner.read_tensor(val.tensor));
                        callback.send(tensor).unwrap();
                    }
                }
            }
        });

        sender
    }
}

impl<B: ReprBackend> Stream<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> Self {
        let sender = Processor::new(runner);
        Self {
            sender,
            queue: Arc::new(Mutex::new(Queue {
                current_index: 0,
                tasks: BTreeMap::new(),
            })),
            _p: PhantomData,
        }
    }

    async fn register_operation(&self, op: RegisterOperation) {
        let mut queue = self.queue.lock().await;
        queue
            .tasks
            .insert(op.index, QueuedTask::RegisterOperation(op));
    }

    async fn register_tensor(&self, op: RegisterTensor) -> TensorDescription {
        let index = op.index;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender
            .send(ProcessorTask::RegisterTensor(op, sender))
            .unwrap();
        let body = rec.recv().unwrap();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(index, QueuedTask::Executed);
        body
    }

    async fn register_tensor_empty(&self, op: RegisterTensorEmpty) -> TensorDescription {
        let index = op.index;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender
            .send(ProcessorTask::RegisterTensorEmpty(op, sender))
            .unwrap();
        let body = rec.recv().unwrap();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(index, QueuedTask::Executed);
        body
    }

    async fn register_orphan(&self, op: RegisterOrphan) {
        let mut queue = self.queue.lock().await;
        queue.tasks.insert(op.index, QueuedTask::RegisterOrphan(op));
    }

    async fn read_tensor(&self, op: ReadTensor) -> TensorData {
        self.dequeue(op.index).await;

        let index = op.index;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender
            .send(ProcessorTask::ReadTensor(op, sender))
            .unwrap();
        let val = rec.recv().unwrap();

        {
            let mut queue = self.queue.lock().await;
            queue.tasks.insert(index, QueuedTask::Executed);
        };

        val
    }

    async fn sync(&self, op: SyncBackend) {
        self.dequeue(op.index).await;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender.send(ProcessorTask::Sync(sender)).unwrap();
        let _val = rec.recv().unwrap();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(op.index, QueuedTask::Executed);
    }

    // End exclude.
    async fn dequeue(&self, end: u64) {
        if end == 0 {
            return;
        }
        loop {
            let mut queue = self.queue.lock().await;
            let key = queue.current_index;

            if key > end - 1 {
                break;
            }

            let task = queue.tasks.remove(&key);
            match task {
                Some(task) => {
                    queue.current_index += 1;
                    match task {
                        QueuedTask::RegisterOperation(op) => {
                            self.sender
                                .send(ProcessorTask::RegisterOperation(op.op))
                                .unwrap();
                        }
                        QueuedTask::RegisterOrphan(val) => self
                            .sender
                            .send(ProcessorTask::RegisterOrphan(val))
                            .unwrap(),
                        QueuedTask::Executed => {}
                    };
                }
                None => {}
            }
        }
    }
}

impl<B: ReprBackend> HttpServer<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    fn new(runner: Runner<B>) -> Self {
        Self { runner }
    }

    async fn start(self, address: &str) {
        tracing_subscriber::fmt::init();

        let app = Router::new()
            .route("/register-operation", post(Self::register_operation))
            .route("/read-tensor", post(Self::read_tensor))
            .route("/register-tensor", post(Self::register_tensor))
            .route("/register-tensor-empty", post(Self::register_tensor_empty))
            .route("/register-orphan", post(Self::register_orphan))
            .route("/sync", post(Self::sync))
            .with_state(Stream::new(self.runner));

        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }

    async fn register_operation(
        State(state): State<Stream<B>>,
        Json(body): Json<RegisterOperation>,
    ) -> (StatusCode, Json<()>) {
        state.register_operation(body).await;
        (StatusCode::CREATED, Json(()))
    }
    async fn read_tensor(
        State(state): State<Stream<B>>,
        Json(body): Json<ReadTensor>,
    ) -> (StatusCode, Json<TensorData>) {
        let data = state.read_tensor(body).await;
        (StatusCode::CREATED, Json(data))
    }
    async fn register_tensor(
        State(state): State<Stream<B>>,
        Json(body): Json<RegisterTensor>,
    ) -> (StatusCode, Json<TensorDescription>) {
        let desc = state.register_tensor(body).await;
        (StatusCode::CREATED, Json(desc))
    }
    async fn register_tensor_empty(
        State(state): State<Stream<B>>,
        Json(body): Json<RegisterTensorEmpty>,
    ) -> (StatusCode, Json<TensorDescription>) {
        let desc = state.register_tensor_empty(body).await;
        (StatusCode::CREATED, Json(desc))
    }
    async fn register_orphan(
        State(state): State<Stream<B>>,
        Json(body): Json<RegisterOrphan>,
    ) -> (StatusCode, Json<()>) {
        state.register_orphan(body).await;
        (StatusCode::CREATED, Json(()))
    }
    async fn sync(
        State(state): State<Stream<B>>,
        Json(body): Json<super::SyncBackend>,
    ) -> (StatusCode, Json<()>) {
        state.sync(body).await;
        (StatusCode::CREATED, Json(()))
    }
}

#[tokio::main]
/// Start a server.
pub async fn start<B: ReprBackend>(device: Device<B>, address: &str)
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    let runner = Runner::<B>::new(device);
    let server = HttpServer::new(runner);

    server.start(address).await
}
