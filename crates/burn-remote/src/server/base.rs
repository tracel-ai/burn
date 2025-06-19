use axum::{
    Router,
    extract::{
        State,
        ws::{self, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::any,
};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::time::Duration;

use burn_ir::{BackendIr, TensorId};
use burn_tensor::Device;
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{filter::filter_fn, registry};

use crate::shared::{ComputeTask, RemoteTensorReq, Task};

use super::session::SessionManager;
use super::tensor_data_service::TensorDataService;

#[derive(Clone)]
pub struct WsServer<B: BackendIr> {
    session_manager: Arc<SessionManager<B>>,
    state: Arc<TensorDataService>,
}

impl<B: BackendIr> WsServer<B> {
    /// Start the server on the given address.
    pub async fn start(device: Device<B>, port: u16) {
        let layer = tracing_subscriber::fmt::layer()
            .with_filter(LevelFilter::INFO)
            .with_filter(filter_fn(|m| {
                if let Some(path) = m.module_path() {
                    // The wgpu crate is logging too much, so we skip `info` level.
                    if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                        return false;
                    }
                }
                true
            }));

        // If we start multiple servers in the same process, this will fail, it's ok
        let _ = registry().with(layer).try_init();

        let address = format!("0.0.0.0:{port}");
        log::info!("Start server {address} on device {device:?}");

        let state = Arc::new(TensorDataService {
            exposed_tensors: tokio::sync::Mutex::new(HashMap::new()),
        });
        let server = Self {
            session_manager: Arc::new(SessionManager::<B>::new(device, state.clone())),
            state,
        };

        // build our application with some routes
        let app = Router::new()
            .route("/response", any(Self::handler_response))
            .route("/request", any(Self::handler_request))
            .route("/data", any(Self::handler_data))
            .with_state(server);

        // run it with hyper
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    }

    async fn handler_response(
        ws: WebSocketUpgrade,
        State(state): State<Self>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| state.handle_socket_response(socket))
    }

    async fn handler_request(ws: WebSocketUpgrade, State(state): State<Self>) -> impl IntoResponse {
        ws.on_upgrade(move |socket| state.handle_socket_request(socket))
    }

    async fn handler_data(ws: WebSocketUpgrade, State(state): State<Self>) -> impl IntoResponse {
        ws.on_upgrade(move |socket| state.handle_socket_data(socket))
    }

    async fn handle_socket_response(self, mut socket: WebSocket) {
        log::info!("[Response Handler] On new connection.");

        let packet = socket.recv().await;
        let msg = match packet {
            Some(msg) => msg,
            None => panic!("Still no message"),
        };

        match msg {
            Ok(ws::Message::Binary(bytes)) => {
                let task = match rmp_serde::from_slice::<Task>(&bytes) {
                    Ok(val) => val,
                    Err(err) => panic!("Only bytes messages are supported {err:?}"),
                };
                let id = match task {
                    Task::Init(id) => id,
                    _ => panic!("Response handler not initialized."),
                };

                let mut receiver = self.session_manager.register_responder(id).await;

                log::info!("Response handler connection active");

                while let Some(mut callback) = receiver.recv().await {
                    let response = callback.recv().await.unwrap();
                    let bytes = rmp_serde::to_vec(&response).unwrap();

                    socket
                        .send(ws::Message::Binary(bytes.into()))
                        .await
                        .unwrap();
                }
            }
            Err(err) => panic!("Can't start the response handler {err:?}"),
            _ => panic!("Unsupported message type"),
        }
    }

    async fn handle_socket_request(self, mut socket: WebSocket) {
        log::info!("[Request Handler] On new connection.");
        let mut session_id = None;

        loop {
            let packet = socket.recv().await;
            let msg = match packet {
                Some(msg) => msg,
                None => {
                    log::info!("Still no message");
                    continue;
                }
            };

            if let Ok(ws::Message::Binary(bytes)) = msg {
                let task = match rmp_serde::from_slice::<Task>(&bytes) {
                    Ok(val) => val,
                    Err(err) => {
                        log::info!("Only bytes message in the json format are supported {err:?}");
                        break;
                    }
                };

                if let Task::Close(id) = task {
                    session_id = Some(id);
                    break;
                }

                let (stream, connection_id, task) =
                    match self.session_manager.stream(&mut session_id, task).await {
                        Some(val) => val,
                        None => {
                            log::info!("Ops session activated {session_id:?}");
                            continue;
                        }
                    };

                match task {
                    ComputeTask::RegisterOperation(op) => {
                        stream.register_operation(op).await;
                    }
                    ComputeTask::RegisterTensor(id, data) => {
                        stream.register_tensor(id, data).await;
                    }
                    ComputeTask::RegisterEmptyTensor(id, shape, dtype) => {
                        stream.register_empty_tensor(id, shape, dtype).await;
                    }
                    ComputeTask::ReadTensor(tensor) => {
                        stream.read_tensor(connection_id, tensor).await;
                    }
                    ComputeTask::SyncBackend => {
                        stream.sync(connection_id).await;
                    }
                    ComputeTask::RegisterTensorRemote(tensor, new_id) => {
                        stream.register_tensor_remote(tensor, new_id).await;
                    }
                    ComputeTask::ExposeTensorRemote { tensor, count } => {
                        stream.expose_tensor_remote(tensor, count).await;
                    }
                }
            } else {
                log::info!("Not a binary message, closing, received {msg:?}");
                break;
            };
        }

        log::info!("Closing session {:?}", session_id);
        self.session_manager.close(session_id).await;
    }

    async fn handle_socket_data(self, mut socket: WebSocket) {
        log::info!("[Data Handler] New connection for download.");

        let packet = socket.recv().await;
        let msg = match packet {
            Some(msg) => msg,
            None => panic!("Still no message"),
        };

        match msg {
            Ok(ws::Message::Binary(bytes)) => {
                let id = match rmp_serde::from_slice::<RemoteTensorReq>(&bytes) {
                    Ok(val) => val.id,
                    Err(err) => panic!("Only bytes messages are supported {err:?}"),
                };

                // Get the requested exposed tensor data
                let bytes: bytes::Bytes = Self::get_exposed_tensor_bytes(id, self.state).await;

                // Send tensor and increment its counter
                socket.send(ws::Message::Binary(bytes)).await.unwrap();
            }
            Err(err) => panic!("Can't start the response handler {err:?}"),
            _ => panic!("Unsupported message type"),
        }
    }

    async fn get_exposed_tensor_bytes(id: TensorId, state: Arc<TensorDataService>) -> bytes::Bytes {
        loop {
            let mut exposed_tensors = state.exposed_tensors.lock().await;
            // take the tensor out of the hashmap while we download
            if let Some(mut exposed_state) = exposed_tensors.remove(&id) {
                exposed_state.cur_download_count += 1;
                if exposed_state.cur_download_count == exposed_state.max_downloads {
                    return exposed_state.bytes;
                } else {
                    let bytes = exposed_state.bytes.clone();
                    exposed_tensors.insert(id, exposed_state);
                    return bytes;
                }
            } else {
                // Tensor not in exposed list: wait a bit
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            core::mem::drop(exposed_tensors);
        }
    }
}

pub async fn start_async<B: BackendIr>(device: Device<B>, port: u16) {
    WsServer::<B>::start(device, port).await;
}

#[tokio::main]
/// Start the server on the given port and [device](Device).
pub async fn start<B: BackendIr>(device: Device<B>, port: u16) {
    start_async::<B>(device, port).await
}
