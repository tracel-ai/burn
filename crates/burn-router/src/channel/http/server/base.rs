use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{ReprBackend, TensorDescription},
    Device, TensorData,
};

use crate::http::{
    CloseConnection, ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor,
    RegisterTensorEmpty, SyncBackend,
};

use super::session::SessionManager;

struct HttpServer<B: ReprBackend> {
    device: Device<B>,
}

impl<B: ReprBackend> HttpServer<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    fn new(device: Device<B>) -> Self {
        Self { device }
    }

    async fn start(self, address: &str) {
        tracing_subscriber::fmt::init();
        let manager = SessionManager::new(self.device);

        let app = Router::new()
            .route("/register-operation", post(Self::register_operation))
            .route("/read-tensor", post(Self::read_tensor))
            .route("/register-tensor", post(Self::register_tensor))
            .route("/register-tensor-empty", post(Self::register_tensor_empty))
            .route("/register-orphan", post(Self::register_orphan))
            .route("/sync", post(Self::sync))
            .route("/close", post(Self::close))
            .with_state(manager);

        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }

    async fn register_operation(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<RegisterOperation>,
    ) -> (StatusCode, Json<()>) {
        let stream = manager.get_stream(body.client_id);
        stream.register_operation(body).await;
        (StatusCode::CREATED, Json(()))
    }
    async fn read_tensor(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<ReadTensor>,
    ) -> (StatusCode, Json<TensorData>) {
        let stream = manager.get_stream(body.client_id);
        let data = stream.read_tensor(body).await;
        (StatusCode::CREATED, Json(data))
    }
    async fn register_tensor(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<RegisterTensor>,
    ) -> (StatusCode, Json<TensorDescription>) {
        let stream = manager.get_stream(body.client_id);
        let desc = stream.register_tensor(body).await;
        (StatusCode::CREATED, Json(desc))
    }
    async fn register_tensor_empty(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<RegisterTensorEmpty>,
    ) -> (StatusCode, Json<TensorDescription>) {
        let stream = manager.get_stream(body.client_id);
        let desc = stream.register_tensor_empty(body).await;
        (StatusCode::CREATED, Json(desc))
    }
    async fn register_orphan(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<RegisterOrphan>,
    ) -> (StatusCode, Json<()>) {
        let stream = manager.get_stream(body.client_id);
        stream.register_orphan(body).await;
        (StatusCode::CREATED, Json(()))
    }
    async fn sync(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<SyncBackend>,
    ) -> (StatusCode, Json<()>) {
        let stream = manager.get_stream(body.client_id);
        stream.sync(body).await;
        (StatusCode::CREATED, Json(()))
    }
    async fn close(
        State(manager): State<SessionManager<B>>,
        Json(body): Json<CloseConnection>,
    ) -> (StatusCode, Json<()>) {
        println!("Closing stream ...");
        let stream = manager.pop_stream(body.client_id);
        stream.close(body).await;
        core::mem::drop(stream);
        println!("Close stream done.");
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
    let server = HttpServer::<B>::new(device);

    server.start(address).await
}
