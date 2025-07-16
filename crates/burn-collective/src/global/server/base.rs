use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::global::{
    server::state::GlobalCollectiveState,
    shared::base::{Message, NodeId},
};
use burn_network::{
    network::{NetworkError, NetworkServer, NetworkStream},
    util::os_shutdown_signal,
};

#[allow(unused)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlobalCollectiveError {
    // Operations that can't be done before registering
    AllReduceBeforeRegister,

    // Can't register a node twice
    MultipleRegister(NodeId),
    // Either a node has unregisterd twice, or a Finish has been called before a Register
    NotRegisteredOnFinish,
    // Finish has been called before a Register operation was finished
    PendingRegisterOnFinish,
    // Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    // Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,

    // First message on socket should be Message::Init
    FirstMsgNotInit,
    // Messages should be rmp_serde serialized `Message` types
    InvalidMessage,
    // A peer behaved unexpectedly
    PeerSentIncoherentTensor,
    // Error from the server
    Server(String),

    // Global Client errors
    // The global collective client received an invalid response
    WrongServerResponse,
    // Client couldn't connect to server
    ServerUnreachable,
}

impl<E: NetworkError> From<E> for GlobalCollectiveError {
    fn from(err: E) -> Self {
        Self::Server(format!("{err:?}"))
    }
}

#[derive(Clone)]
pub struct GlobalCollectiveServer {
    state: Arc<Mutex<GlobalCollectiveState>>,
}

impl GlobalCollectiveServer {
    pub(crate) async fn start<F, S: NetworkServer + Debug>(
        shutdown_signal: F,
        port: u16,
    ) -> Result<(), GlobalCollectiveError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let state = GlobalCollectiveState::new();
        let server = Self {
            state: Arc::new(tokio::sync::Mutex::new(state)),
        };

        S::new(port)
            .route("/response", {
                let server = server.clone();
                async move |socket| {
                    if let Err(err) = server.handle_socket_response::<S>(socket).await {
                        log::error!("[Response Handler] Error: {err:?}")
                    }
                }
            })
            .route("/request", {
                let server = server.clone();
                async move |socket| {
                    if let Err(err) = server.handle_socket_request::<S>(socket).await {
                        log::error!("[Request Handler] Error: {err:?}")
                    }
                }
            })
            .serve(shutdown_signal)
            .await
            .map_err(|err| GlobalCollectiveError::Server(format!("{err:?}")))?;

        Ok(())
    }

    async fn handle_socket_response<S: NetworkServer>(
        self,
        mut stream: S::Stream,
    ) -> Result<(), GlobalCollectiveError> {
        log::info!("[Response Handler] On new connection.");

        let msg = stream
            .recv()
            .await
            .map_err(|err| GlobalCollectiveError::Server(format!("{err:?}")))?;
        let Some(msg) = msg else {
            log::warn!("Response socket closed early!");
            return Ok(());
        };

        let msg = rmp_serde::from_slice::<Message>(&msg.data)
            .map_err(|_| GlobalCollectiveError::InvalidMessage)?;

        let Message::Init(id) = msg else {
            return Err(GlobalCollectiveError::FirstMsgNotInit);
        };

        let mut receiver = {
            let mut state = self.state.lock().await;
            state.get_session_responder(id)
        };

        while let Some(response) = receiver.recv().await {
            let bytes = rmp_serde::to_vec(&response).unwrap();

            stream.send(bytes.into()).await?;
        }

        log::info!("[Response Handler] Closing connection.");
        Ok(())
    }

    async fn handle_socket_request<S: NetworkServer>(
        self,
        mut stream: S::Stream,
    ) -> Result<(), GlobalCollectiveError> {
        log::info!("[Request Handler] On new connection.");

        let mut session_id = None;

        loop {
            let packet = stream.recv().await?;
            let Some(msg) = packet else {
                log::info!("Peer closed the connection");
                break;
            };

            let mut state = self.state.lock().await;

            let msg = rmp_serde::from_slice::<Message>(&msg.data)
                .map_err(|_| GlobalCollectiveError::InvalidMessage)?;
            match msg {
                Message::Init(id) => {
                    state.init_session(id);
                    session_id = Some(id);
                }
                Message::Request(request_id, remote_request) => {
                    let session_id = session_id.ok_or(GlobalCollectiveError::FirstMsgNotInit)?;
                    state.process(session_id, request_id, remote_request).await;
                }
            }
        }

        Ok(())
    }
}

/// Start the server on the given port
pub async fn start<S: NetworkServer + Debug>(port: u16) {
    let res = GlobalCollectiveServer::start::<_, S>(os_shutdown_signal(), port).await;
    if let Err(err) = res {
        eprintln!("Global Collective Server error: {err:?}");
    }
}
