use std::{net::SocketAddr, sync::Arc};

use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{self, WebSocket},
    },
    response::IntoResponse,
    routing::any,
};
use tokio::sync::Mutex;
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::{
    Layer, filter::filter_fn, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

use crate::global::{server::state::GlobalCollectiveState, shared::Message};

#[derive(Clone)]
pub struct GlobalCollectiveServer {
    state: Arc<Mutex<GlobalCollectiveState>>,
}

fn init_logging() {
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
}

impl GlobalCollectiveServer {
    pub(crate) async fn start<F>(shutdown_signal: F, port: u16)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        init_logging();

        let address = format!("0.0.0.0:{port}");
        log::info!("Start server {address}");

        let state = GlobalCollectiveState::new();

        let server = Self {
            state: Arc::new(tokio::sync::Mutex::new(state)),
        };

        // build our application with some routes
        let app = Router::new()
            .route("/response", any(Self::handler_response))
            .route("/request", any(Self::handler_request))
            .with_state(server);

        // run it with hyper
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown_signal)
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
            None => panic!("Still no message"),
        };

        match msg {
            Ok(ws::Message::Binary(bytes)) => {
                let id = match rmp_serde::from_slice::<Message>(&bytes) {
                    Ok(val) => match val {
                        Message::Init(id) => id,
                        _ => panic!("First message on /response should be a register"),
                    },
                    Err(err) => panic!("Only bytes messages are supported {err:?}"),
                };

                let mut receiver = {
                    let mut state = self.state.lock().await;
                    state.get_session_responder(id)
                };

                while let Some(response) = receiver.recv().await {
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
                let mut state = self.state.lock().await;

                match rmp_serde::from_slice::<Message>(&bytes) {
                    Ok(val) => match val {
                        Message::Init(id) => {
                            state.init_session(id);
                            session_id = Some(id);
                        }
                        Message::Request(request_id, remote_request) => {
                            let session_id = session_id
                                .expect("Must init session before requesting operations!");
                            state.process(session_id, request_id, remote_request).await;
                        }
                    },
                    Err(err) => {
                        log::info!("Only bytes message in the json format are supported {err:?}");
                        break;
                    }
                };
            } else {
                log::info!("Not a binary message, closing, received {msg:?}");
                break;
            };
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

/// Start the server on the given port
pub async fn start(port: u16) {
    GlobalCollectiveServer::start(shutdown_signal(), port).await;
}
