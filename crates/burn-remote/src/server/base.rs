use burn_communication::{
    CommunicationChannel, Message, Protocol, ProtocolServer,
    data_service::{TensorDataServer, TensorDataService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use burn_ir::BackendIr;
use burn_tensor::Device;

use crate::shared::{ComputeTask, Task};

use super::session::SessionManager;

pub struct RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    _b: PhantomData<B>,
    _n: PhantomData<P>,
}

impl<B, P> RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Start the server on the given address.
    pub async fn start(device: Device<B>, server: P::Server) {
        let cancel_token = CancellationToken::new();
        let data_service = Arc::new(TensorDataService::<B, P>::new(cancel_token));
        let session_manager = Arc::new(SessionManager::<B, P>::new(device, data_service.clone()));

        let _server = server
            .route("/response", {
                let session_manager = session_manager.clone();
                move |stream| Self::handle_socket_response(session_manager, stream)
            })
            .route("/request", {
                let session_manager = session_manager.clone();
                move |stream| Self::handle_socket_request(session_manager, stream)
            })
            .route_tensor_data_service(data_service)
            .serve(os_shutdown_signal())
            .await;
    }

    async fn handle_socket_response(
        session_manager: Arc<SessionManager<B, P>>,
        mut socket: <P::Server as ProtocolServer>::Channel,
    ) {
        log::info!("[Response Handler] On new connection.");

        let packet = socket.recv().await;
        let msg = match packet {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                log::info!("Response stream closed");
                return;
            }
            Err(e) => {
                log::info!("Response stream error on init: {e:?}");
                return;
            }
        };

        let id = match rmp_serde::from_slice::<Task>(&msg.data) {
            Ok(Task::Init(session_id)) => session_id,
            msg => {
                log::error!("Message is not a valid initialization task {msg:?}");
                return;
            }
        };

        let mut receiver = session_manager.register_responder(id).await;

        log::info!("Response handler connection active");

        while let Some(mut callback) = receiver.recv().await {
            let response = callback.recv().await.unwrap();
            let bytes = rmp_serde::to_vec(&response).unwrap();

            socket.send(Message::new(bytes.into())).await.unwrap();
        }
    }

    async fn handle_socket_request(
        session_manager: Arc<SessionManager<B, P>>,
        mut socket: <P::Server as ProtocolServer>::Channel,
    ) {
        log::info!("[Request Handler] On new connection.");
        let mut session_id = None;

        loop {
            let packet = socket.recv().await;
            let msg = match packet {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    log::info!("Request stream closed");
                    break;
                }
                Err(e) => {
                    log::info!("Request stream error: {e:?}, Closing.");
                    break;
                }
            };

            let task = match rmp_serde::from_slice::<Task>(&msg.data) {
                Ok(val) => val,
                Err(err) => {
                    log::info!("Only bytes message in the json format are supported {err:?}");
                    break;
                }
            };

            if let Task::Close(id) = task {
                session_id = Some(id);
                break;
            }

            let (stream, connection_id, task) =
                match session_manager.stream(&mut session_id, task).await {
                    Some(val) => val,
                    None => {
                        log::info!("Ops session activated {session_id:?}");
                        continue;
                    }
                };

            match task {
                ComputeTask::RegisterOperation(op) => {
                    stream.register_operation(op).await;
                }
                ComputeTask::RegisterTensor(id, data) => {
                    stream.register_tensor(id, data).await;
                }
                ComputeTask::ReadTensor(tensor) => {
                    stream.read_tensor(connection_id, tensor).await;
                }
                ComputeTask::SyncBackend => {
                    stream.sync(connection_id).await;
                }
                ComputeTask::RegisterTensorRemote(tensor, new_id) => {
                    stream.register_tensor_remote(tensor, new_id).await;
                }
                ComputeTask::ExposeTensorRemote {
                    tensor,
                    count,
                    transfer_id,
                } => {
                    stream
                        .expose_tensor_remote(tensor, count, transfer_id)
                        .await;
                }
                ComputeTask::Seed(seed) => {
                    stream.seed(seed).await;
                }
                ComputeTask::SupportsDType(dtype) => {
                    stream.supports_dtype(connection_id, dtype).await
                }
            }
        }

        log::info!("Closing session {session_id:?}");
        session_manager.close(session_id).await;
    }
}

/// Start the server on the given port and [device](Device).
pub async fn start_websocket_async<B: BackendIr>(device: Device<B>, port: u16) {
    let server = WsServer::new(port);
    RemoteServer::<B, WebSocket>::start(device, server).await;
}

#[tokio::main]
/// Start the server on the given port and [device](Device).
pub async fn start_websocket<B: BackendIr>(device: Device<B>, port: u16) {
    start_websocket_async::<B>(device, port).await;
}
