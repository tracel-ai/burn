use std::{collections::HashMap, marker::PhantomData, net::SocketAddr};

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

use crate::{
    server::stream::Stream,
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
        let address = format!("0.0.0.0:{port}");
        println!("Start server {address} on device {device:?}");

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
        let runner = Runner::new(device);
        let mut streams = HashMap::<u64, Stream<B>>::new();

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

                let stream = match streams.get(&task.id.stream_id) {
                    Some(stream) => {
                        // Only when necessary.
                        stream.flush(task.id);
                        stream
                    },
                    None => {
                        let stream = Stream::<B>::new(runner.clone());
                        streams.insert(task.id.stream_id, stream);
                        streams.get(&task.id.stream_id).unwrap()
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
                        let response = stream.read_tensor(task.id, tensor);
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::SyncBackend => {
                        let response = stream.sync(task.id);
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    TaskContent::FlushBackend => {
                        let response = stream.flush(task.id);
                        let bytes = rmp_serde::to_vec(&response).unwrap();
                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                }
            } else {
                println!("Not a binary message, closing, received {msg:?}");
                break;
            };
        }

        println!("Closing connection");
        for (id, stream) in streams.drain() {
            println!("Closing stream {id}");
            stream.close();
        }
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
