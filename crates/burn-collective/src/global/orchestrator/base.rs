use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::global::{
    orchestrator::state::GlobalCollectiveState,
    shared::{CollectiveMessage, GlobalCollectiveError},
};
use burn_communication::{
    CommunicationChannel, Message, ProtocolServer, util::os_shutdown_signal, websocket::WsServer,
};

/// The global collective state manages collective operations on the global level
#[derive(Clone)]
pub(crate) struct GlobalOrchestrator {
    state: Arc<Mutex<GlobalCollectiveState>>,
}

impl GlobalOrchestrator {
    /// Starts the comms server with two routes: "/request" and "/response"
    pub(crate) async fn start<F, S: ProtocolServer + Debug>(
        shutdown_signal: F,
        comms_server: S,
    ) -> Result<(), GlobalCollectiveError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let state = GlobalCollectiveState::new();
        let server = Self {
            state: Arc::new(tokio::sync::Mutex::new(state)),
        };

        comms_server
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

    async fn handle_socket_response<S: ProtocolServer>(
        self,
        mut stream: S::Channel,
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

        let msg = rmp_serde::from_slice::<CollectiveMessage>(&msg.data)
            .map_err(|_| GlobalCollectiveError::InvalidMessage)?;

        let CollectiveMessage::Init(id) = msg else {
            return Err(GlobalCollectiveError::FirstMsgNotInit);
        };

        let mut receiver = {
            let mut state = self.state.lock().await;
            state.get_session_responder(id)
        };

        while let Some(response) = receiver.recv().await {
            let bytes = rmp_serde::to_vec(&response).unwrap();

            stream.send(Message::new(bytes.into())).await?;
        }

        log::info!("[Response Handler] Closing connection.");
        Ok(())
    }

    async fn handle_socket_request<S: ProtocolServer>(
        self,
        mut stream: S::Channel,
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

            let msg = rmp_serde::from_slice::<CollectiveMessage>(&msg.data)
                .map_err(|_| GlobalCollectiveError::InvalidMessage)?;
            match msg {
                CollectiveMessage::Init(id) => {
                    state.init_session(id);
                    session_id = Some(id);
                }
                CollectiveMessage::Request(request_id, remote_request) => {
                    let session_id = session_id.ok_or(GlobalCollectiveError::FirstMsgNotInit)?;
                    state
                        .process_request(session_id, request_id, remote_request)
                        .await;
                }
            }
        }

        Ok(())
    }
}

/// Start a global orchestrator with WebSocket on the given port
pub async fn start_global_orchestrator(port: u16) {
    let server = WsServer::new(port);
    let res = GlobalOrchestrator::start(os_shutdown_signal(), server).await;
    if let Err(err) = res {
        log::error!("Global Collective Orchestrator error: {err:?}");
    }
}
