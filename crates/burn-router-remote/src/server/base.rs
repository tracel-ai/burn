use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{mpsc::Receiver, Arc},
};

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
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{filter::filter_fn, registry};

use crate::{
    server::stream::StreamManager,
    shared::{Task, TaskContent, TaskResponse},
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

    pub async fn register_writer(&self, session_id: u64) -> Receiver<Receiver<TaskResponse>> {
        log::info!("Register writer {session_id}");
        let mut sessions = self.sessions.lock().await;
        self.register_session(&mut sessions, session_id);

        let session = sessions.get_mut(&session_id).unwrap();
        session.init_writer()
    }

    fn register_session(&self, sessions: &mut HashMap<u64, StreamManager<B>>, id: u64) {
        if !sessions.contains_key(&id) {
            log::info!("Create session {id}");
            let session = StreamManager::new(self.runner.clone());
            sessions.insert(id, session);
        }
    }
    pub async fn stream(&self, session_id: &mut Option<u64>, task: &Task) -> Option<Stream<B>> {
        let mut sessions = self.sessions.lock().await;

        let session_id = match session_id {
            Some(id) => *id,
            None => match task.content {
                TaskContent::Init(id) => {
                    log::info!("Init session receiver {id}");
                    *session_id = Some(id);
                    self.register_session(&mut sessions, id);
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

        let sessions = SessionManager::<B>::new(device);
        let state = Self {
            sessions: Arc::new(sessions),
        };

        // build our application with some routes
        let app = Router::new()
            .route("/load", any(Self::ws_load_handler))
            .route("/ops", any(Self::ws_ops_handler))
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

    async fn ws_load_handler(
        ws: WebSocketUpgrade,
        State(session): State<Self>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| session.handle_socket_load(socket))
    }
    async fn ws_ops_handler(
        ws: WebSocketUpgrade,
        State(session): State<Self>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| session.handle_socket_ops(socket))
    }

    async fn handle_socket_load(self, mut socket: WebSocket) {
        log::info!("On new load connection.");

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

            let receiver = self.sessions.register_writer(id).await;

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

    async fn handle_socket_ops(self, mut socket: WebSocket) {
        log::info!("On new ops connection.");
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
