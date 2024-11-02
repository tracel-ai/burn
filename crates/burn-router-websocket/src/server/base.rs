use std::{marker::PhantomData, net::SocketAddr};

use axum::{
    extract::ws::{self, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::any,
    Router,
};

use burn_router::Runner;
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::ReprBackend,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

//allows to extract the IP of connecting user
use axum::extract::connect_info::ConnectInfo;

use crate::shared::{Task, TaskContent};

use super::processor::{Processor, ProcessorTask};

pub struct WebSocketServer<B: ReprBackend> {
    _p: PhantomData<B>,
}

impl<B: ReprBackend> WebSocketServer<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    /// Start the server on the given address.
    pub async fn start(address: &str) {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                    format!("{}=debug,tower_http=debug", env!("CARGO_CRATE_NAME")).into()
                }),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();

        // build our application with some routes
        let app = Router::new().route("/ws", any(Self::ws_handler));

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
        ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| Self::handle_socket(socket, addr))
    }

    /// Actual websocket statemachine (one will be spawned per connection)
    async fn handle_socket(mut socket: WebSocket, _who: SocketAddr) {
        let processor = Processor::new(Runner::<B>::new(Default::default()));

        loop {
            let packet = socket.recv().await;
            let msg = match packet {
                Some(msg) => msg,
                None => {
                    println!("Still no message");
                    continue;
                }
            };

            if let Ok(ws::Message::Binary(bytes)) = msg {
                let task = match rmp_serde::from_slice::<Task>(&bytes) {
                    Ok(val) => val,
                    Err(err) => {
                        println!("Only bytes message in the json format are supported {err:?}");
                        break;
                    }
                };

                match task.content {
                    TaskContent::RegisterOperation(op) => processor
                        .send(ProcessorTask::RegisterOperation(op))
                        .unwrap(),
                    TaskContent::RegisterTensor(data) => {
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::RegisterTensor(task.id, data, sender))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::RegisterTensorEmpty(shape, dtype) => {
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::RegisterTensorEmpty(
                                task.id, shape, dtype, sender,
                            ))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::RegisterOrphan(id) => {
                        processor.send(ProcessorTask::RegisterOrphan(id)).unwrap()
                    }
                    TaskContent::ReadTensor(tensor) => {
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::ReadTensor(task.id, tensor, sender))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::SyncBackend => {
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::Sync(task.id, sender))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                }
            } else {
                println!("Not a binary message, closing, received {msg:?}");
                break;
            }
        }
        processor.send(ProcessorTask::Close).unwrap();
    }
}
