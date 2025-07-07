use std::sync::Arc;
use tokio::sync::Mutex;

use crate::global::{server::state::GlobalCollectiveState, shared::base::Message};
use burn_network::{
    network::{NetworkServer, NetworkStream},
    websocket::WsServer,
};

#[derive(Clone)]
pub struct GlobalCollectiveServer {
    state: Arc<Mutex<GlobalCollectiveState>>,
}

impl GlobalCollectiveServer {
    pub(crate) async fn start<F, S: NetworkServer<State = Self>>(shutdown_signal: F, port: u16)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let state = GlobalCollectiveState::new();
        let server = Self {
            state: Arc::new(tokio::sync::Mutex::new(state)),
        };

        S::new(port)
            .route("/response", async |state, socket| {
                state
                    .handle_socket_response::<S::ServerStream>(socket)
                    .await
            })
            .route("/request", async |state, socket| {
                state.handle_socket_request::<S::ServerStream>(socket).await
            })
            .serve(server, shutdown_signal)
            .await;
    }

    async fn handle_socket_response<S: NetworkStream>(self, mut stream: S) {
        log::info!("[Response Handler] On new connection.");

        let msg = stream.recv().await;
        let Ok(Some(msg)) = msg else {
            panic!("Expected a Init message on /response");
        };

        let id = match rmp_serde::from_slice::<Message>(&msg.data) {
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

            stream.send(bytes.into()).await.unwrap();
        }
    }

    async fn handle_socket_request<S: NetworkStream>(self, mut socket: S) {
        log::info!("[Request Handler] On new connection.");

        let mut session_id = None;

        loop {
            let packet = socket.recv().await;
            let msg = match packet {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    log::info!("Peer closed the connection");
                    break;
                }
                Err(err) => {
                    panic!("Failed to receive message from websocket: {err:?}");
                }
            };

            let mut state = self.state.lock().await;

            match rmp_serde::from_slice::<Message>(&msg.data) {
                Ok(val) => match val {
                    Message::Init(id) => {
                        state.init_session(id);
                        session_id = Some(id);
                    }
                    Message::Request(request_id, remote_request) => {
                        let session_id =
                            session_id.expect("Must init session before requesting operations!");
                        state.process(session_id, request_id, remote_request).await;
                    }
                },
                Err(err) => {
                    panic!("Invalid message format, must be msgpack: {err:?}");
                }
            };
        }
    }
}

/// Start the server on the given port
pub async fn start(port: u16) {
    type Server = WsServer<GlobalCollectiveServer>;
    GlobalCollectiveServer::start::<_, Server>(burn_network::util::os_shutdown_signal(), port)
        .await;
}
