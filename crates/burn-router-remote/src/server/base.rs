use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use axum::{
    extract::{
        ws::{self, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::any,
    Router,
};

use burn_router::Runner;
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::ReprBackend,
    Device,
};
use tokio::sync::Mutex;
use tracing_core::LevelFilter;

use crate::{
    server::stream::StreamManager,
    shared::{Task, TaskContent},
};

use super::stream::Stream;

#[derive(Clone)]
pub struct WsServer<B: ReprBackend> {
    sessions: Arc<SessionManager<B>>,
}

pub struct SessionManager<B: ReprBackend> {
    runner: Runner<B>,
    sessions: tokio::sync::Mutex<HashMap<u64, StreamManager<B>>>,
}

impl<B: ReprBackend> SessionManager<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(device: Device<B>) -> Self {
        Self {
            runner: Runner::new(device),
            sessions: Mutex::new(Default::default()),
        }
    }

    pub async fn stream(&self, session_id: &mut Option<u64>, task: &Task) -> Option<Stream<B>> {
        let mut sessions = self.sessions.lock().await;

        let session_id = match session_id {
            Some(id) => *id,
            None => match task.content {
                TaskContent::Init(id) => {
                    *session_id = Some(id);
                    if !sessions.contains_key(&id) {
                        let session = StreamManager::new(self.runner.clone());
                        sessions.insert(id, session);
                    }
                    return None;
                }
                _ => panic!("The first message should be init the session"),
            },
        };

        match sessions.get_mut(&session_id) {
            Some(session) => Some(session.select(task)),
            None => {
                panic!("To be initialized");
            }
        }
    }

    pub async fn close(&self, session_id: Option<u64>) {
        if let Some(id) = session_id {
            let mut sessions = self.sessions.lock().await;
            if let Some(session) = sessions.get_mut(&id) {
                session.close();
            }
        }
    }
}

impl<B: ReprBackend> WsServer<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    /// Start the server on the given address.
    pub async fn start(device: Device<B>, port: u16) {
        tracing_subscriber::fmt()
            .with_max_level(LevelFilter::DEBUG)
            .init();

        let address = format!("0.0.0.0:{port}");
        log::info!("Start server {address} on device {device:?}");

        let sessions = SessionManager::<B>::new(device);
        let state = Self {
            sessions: Arc::new(sessions),
        };

        // build our application with some routes
        let app = Router::new()
            .route("/ws", any(Self::ws_handler))
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

    async fn ws_handler(ws: WebSocketUpgrade, State(session): State<Self>) -> impl IntoResponse {
        ws.on_upgrade(move |socket| session.handle_socket(socket))
    }

    async fn handle_socket(self, mut socket: WebSocket) {
        log::info!("On new connection");
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

                let stream = match self.sessions.stream(&mut session_id, &task).await {
                    Some(val) => val,
                    None => continue,
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
                        let id = task.id.clone();
                        let response = stream.read_tensor(id, tensor);
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::SyncBackend => {
                        let id = task.id.clone();
                        let response = stream.sync(id);
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::FlushBackend => {
                        let id = task.id.clone();
                        let response = stream.flush(id);
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::Init(_) => {}
                }
            } else {
                log::info!("Not a binary message, closing, received {msg:?}");
                break;
            };
        }

        log::info!("Closing connection");
        self.sessions.close(session_id).await;
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
