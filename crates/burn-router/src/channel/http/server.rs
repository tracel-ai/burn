use std::{collections::BTreeMap, sync::Arc};

use crate::{Runner, RunnerClient};
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{ReprBackend, TensorDescription},
    Device, TensorData,
};
use tokio::sync::Mutex;

use super::{
    ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor, RegisterTensorEmpty, SyncBackend,
};

struct HttpServer<B: ReprBackend> {
    runner: Runner<B>,
}

enum Task {
    RegisterOperation(RegisterOperation),
    RegisterOrphan(RegisterOrphan),
    Executed,
}

#[derive(Clone)]
struct Stream<B: ReprBackend> {
    runner: Runner<B>,
    queue: Arc<Mutex<Queue>>,
}

struct Queue {
    current_index: u64,
    tasks: BTreeMap<u64, Task>,
}

impl<B: ReprBackend> Stream<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> Self {
        Self {
            runner,
            queue: Arc::new(Mutex::new(Queue {
                current_index: 0,
                tasks: BTreeMap::new(),
            })),
        }
    }

    async fn register_operation(&self, op: RegisterOperation) {
        println!("Register operation {}.", op.index);
        let index = op.index;

        {
            let mut queue = self.queue.lock().await;
            queue.tasks.insert(op.index, Task::RegisterOperation(op));
        };

        self.dequeue(index + 1).await;
    }

    async fn register_tensor(&self, op: RegisterTensor) -> TensorDescription {
        println!("Register tensor {}.", op.index);
        let body = self.runner.register_tensor_data_desc(op.data);
        let mut queue = self.queue.lock().await;
        queue.tasks.insert(op.index, Task::Executed);
        body
    }

    async fn register_tensor_empty(&self, op: RegisterTensorEmpty) -> TensorDescription {
        println!("Register tensor empty {}.", op.index);

        let body = self.runner.register_empty_tensor_desc(op.shape, op.dtype);
        let mut queue = self.queue.lock().await;
        queue.tasks.insert(op.index, Task::Executed);
        body
    }

    async fn register_orphan(&self, op: RegisterOrphan) {
        println!("Register orphan {}.", op.index);
        let index = op.index;

        {
            let mut queue = self.queue.lock().await;
            queue.tasks.insert(op.index, Task::RegisterOrphan(op));
        };

        self.dequeue(index + 1).await;
    }

    async fn read_tensor(&self, op: ReadTensor) -> TensorData {
        println!("Register read {}.", op.index);
        self.dequeue(op.index).await;

        let val = {
            let val = self.runner.read_tensor(op.tensor).await;

            let mut queue = self.queue.lock().await;
            queue.tasks.insert(op.index, Task::Executed);
            val
        };

        val
    }

    async fn sync(&self, op: SyncBackend) {
        println!("Register sync {}.", op.index);

        self.dequeue(op.index).await;
        self.runner.sync();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(op.index, Task::Executed);
    }

    // End exclude.
    async fn dequeue(&self, end: u64) {
        if end == 0 {
            return;
        }
        loop {
            let mut queue = self.queue.lock().await;
            let key = queue.current_index;
            // println!("Processing key {key:?}...");

            if key > end - 1 {
                break;
            }

            let task = queue.tasks.remove(&key);
            match task {
                Some(task) => {
                    queue.current_index += 1;
                    match task {
                        Task::RegisterOperation(op) => {
                            println!("Processed operation lazy {key:?}.");
                            self.runner.register(op.op);
                        }
                        Task::RegisterOrphan(val) => {
                            println!("Processed orphan lazy {key:?}.");
                            self.runner.register_orphan(&val.id);
                        }
                        Task::Executed => {
                            println!("Already executed {key:?}.");
                        }
                    };
                }
                None => {
                    println!("Key {key:?} not found. waiting ...");
                }
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
        println!("Start server ...");
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
