use std::{marker::PhantomData, net::SocketAddr};

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
use tracing_core::LevelFilter;

use crate::{
    server::stream::StreamManager,
    shared::{Task, TaskContent},
};

pub struct WsServer<B: ReprBackend> {
    _p: PhantomData<B>,
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

        // build our application with some routes
        let app = Router::new()
            .route("/ws", any(Self::ws_handler))
            .with_state(device);

        // run it with hyper
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    }

    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(device): State<Device<B>>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| Self::handle_socket(device, socket))
    }

    async fn handle_socket(device: Device<B>, mut socket: WebSocket) {
        log::info!("On new connection");
        let runner = Runner::new(device);

        let mut streams = StreamManager::<B>::new(runner);

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
                let stream = streams.select(&task);

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
                }
            } else {
                log::info!("Not a binary message, closing, received {msg:?}");
                break;
            };
        }

        log::info!("Closing connection");
        streams.close();
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
