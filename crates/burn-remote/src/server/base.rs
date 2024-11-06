use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::{
        ws::{self, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::any,
    Router,
};

use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::ReprBackend,
    Device,
};
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{filter::filter_fn, registry};

use crate::shared::{Task, TaskContent};

use super::session::SessionManager;

#[derive(Clone)]
pub struct WsServer<B: ReprBackend> {
    state: Arc<SessionManager<B>>,
}

impl<B: ReprBackend> WsServer<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
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
        registry().with(layer).init();

        let address = format!("0.0.0.0:{port}");
        log::info!("Start server {address} on device {device:?}");

        let state = SessionManager::<B>::new(device);
        let state = Self {
            state: Arc::new(state),
        };

        // build our application with some routes
        let app = Router::new()
            .route("/response", any(Self::handler_response))
            .route("/request", any(Self::handler_request))
            .with_state(state);

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

    async fn handle_socket_response(self, mut socket: WebSocket) {
        log::info!("[Response Handler] On new connection.");

        let packet = socket.recv().await;
        let msg = match packet {
            Some(msg) => msg,
            None => {
                log::info!("Still no message");
                panic!("");
            }
        };

        if let Ok(ws::Message::Binary(bytes)) = msg {
            let task = match rmp_serde::from_slice::<Task>(&bytes) {
                Ok(val) => val,
                Err(err) => {
                    log::info!("Only bytes message in the json format are supported {err:?}");
                    panic!("");
                }
            };
            let id = match task.content {
                TaskContent::Init(id) => id,
                _ => panic!(""),
            };

            let receiver = self.state.register_responder(id).await;

            log::info!("Load connection activated.");
            let handler = tokio::runtime::Handle::current();

            // Without the thread we might deadlock with tokio.
            std::thread::spawn(move || {
                while let Ok(callback) = receiver.recv() {
                    let response = callback.recv().unwrap();
                    let bytes = rmp_serde::to_vec(&response).unwrap();

                    handler
                        .block_on(async { socket.send(ws::Message::Binary(bytes)).await.unwrap() });
                }
            });
        } else {
            panic!("");
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

                let stream = match self.state.stream(&mut session_id, &task).await {
                    Some(val) => val,
                    None => {
                        log::info!("Ops session activated {session_id:?}");
                        continue;
                    }
                };

                match task.content {
                    TaskContent::RegisterOperation(op) => {
                        stream.register_operation(op);
                    }
                    TaskContent::RegisterTensor(id, data) => {
                        stream.register_tensor(id, data);
                    }
                    TaskContent::RegisterOrphan(id) => {
                        stream.register_orphan(id);
                    }
                    TaskContent::ReadTensor(tensor) => {
                        stream.read_tensor(task.id, tensor);
                    }
                    TaskContent::SyncBackend => {
                        stream.sync(task.id);
                    }
                    TaskContent::FlushBackend => {
                        stream.flush(task.id);
                    }
                    TaskContent::Init(_) => {}
                }
            } else {
                log::info!("Not a binary message, closing, received {msg:?}");
                break;
            };
        }

        log::info!("Closing connection");
        self.state.close(session_id).await;
    }
}

#[tokio::main(flavor = "current_thread")]
/// Start the server on the given port and [device](Device).
pub async fn start<B: ReprBackend>(device: Device<B>, port: u16)
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    WsServer::<B>::start(device, port).await;
}
