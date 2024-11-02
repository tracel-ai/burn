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
//allows to extract the IP of connecting user
use axum::extract::connect_info::ConnectInfo;

use crate::shared::{Task, TaskContent};

use super::processor::{Processor, ProcessorTask};

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
    pub async fn start(device: Device<B>, address: &str) {
        println!("Start server {address} on device {device:?}");
        // tracing_subscriber::registry()
        //     .with(
        //         tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        //             format!("{}=debug,tower_http=debug", env!("CARGO_CRATE_NAME")).into()
        //         }),
        //     )
        //     .with(tracing_subscriber::fmt::layer())
        //     .init();

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
        ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| Self::handle_socket(device, socket, addr))
    }

    /// Actual websocket statemachine (one will be spawned per connection)
    async fn handle_socket(device: Device<B>, mut socket: WebSocket, _who: SocketAddr) {
        println!("On new connection");
        let processor = Processor::new(Runner::<B>::new(device));

        loop {
            let packet = socket.recv().await;
            let start = std::time::Instant::now();
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
                    TaskContent::RegisterOperation(op) => {
                        processor
                            .send(ProcessorTask::RegisterOperation(op))
                            .unwrap();
                        println!("Register Operation {:?} {:?}", task.id, start.elapsed());
                    }
                    TaskContent::RegisterTensor(data) => {
                        let start = std::time::Instant::now();
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::RegisterTensor(task.id, data, sender))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                        println!("Register Tensor {:?} {:?}", task.id, start.elapsed());
                    }
                    TaskContent::RegisterTensorEmpty(shape, dtype) => {
                        let start = std::time::Instant::now();
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::RegisterTensorEmpty(
                                task.id, shape, dtype, sender,
                            ))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                        println!("Register Tensor Empty {:?} {:?}", task.id, start.elapsed());
                    }
                    TaskContent::RegisterOrphan(id) => {
                        let start = std::time::Instant::now();
                        processor.send(ProcessorTask::RegisterOrphan(id)).unwrap();
                        println!("Register Orphan {:?} {:?}", task.id, start.elapsed());
                    }
                    TaskContent::ReadTensor(tensor) => {
                        let start = std::time::Instant::now();
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::ReadTensor(task.id, tensor, sender))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                        println!("Read Tensor {:?} {:?}", task.id, start.elapsed());
                    }
                    TaskContent::SyncBackend => {
                        let start = std::time::Instant::now();
                        let (sender, recv) = std::sync::mpsc::channel();

                        processor
                            .send(ProcessorTask::Sync(task.id, sender))
                            .unwrap();

                        let response = recv.recv().expect("A callback");
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                        println!("Sync Backend {:?} {:?}", task.id, start.elapsed());
                    }
                }
            } else {
                println!("Not a binary message, closing, received {msg:?}");
                break;
            };

            println!("Took {:?}", start.elapsed());
        }
        println!("Closing connection");
        processor.send(ProcessorTask::Close).unwrap();
    }
}

#[tokio::main(flavor = "current_thread")]
/// Start a server.
pub async fn start<B: ReprBackend>(device: Device<B>, address: &str)
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    WsServer::<B>::start(device, address).await;
}
