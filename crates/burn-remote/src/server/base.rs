use burn_communication::{
    CommunicationChannel, Message, Protocol, ProtocolServer,
    data_service::{TensorDataServer, TensorDataService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::RunnerClient;
use burn_std::id::StreamId;

use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponse, TaskResponseContent};

use super::session::SessionManager;

/// HTTP-style server for the burn-remote protocol.
///
/// Compute tasks are processed **inline** on the request-handling task: there is no
/// per-stream worker thread and no compute-side mpsc channel. The
/// [`SessionManager`] only owns the per-session response queue (one `mpsc::Sender` per
/// session, drained by the response-writing task). The client-side stream id riding on
/// each [`ConnectionId`] is threaded through to the runner via [`StreamId::executes`], so
/// stream-aware backends (fusion, etc.) see the right stream context per op.
pub struct RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    _b: PhantomData<B>,
    _p: PhantomData<P>,
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

        // Read the init handshake to learn which session this responder belongs to.
        let msg = match socket.recv().await {
            Ok(Some(m)) => m,
            Ok(None) => {
                log::info!("Response stream closed before init handshake");
                return;
            }
            Err(err) => {
                log::warn!("Response stream error during init handshake: {err:?}");
                return;
            }
        };

        let session_id = match rmp_serde::from_slice::<Task>(&msg.data) {
            Ok(Task::Init(id)) => id,
            Ok(other) => {
                log::error!(
                    "Response handshake expected Task::Init, got {other:?}; closing stream"
                );
                return;
            }
            Err(err) => {
                log::error!("Failed to decode response init handshake: {err:?}");
                return;
            }
        };

        log::info!("Init responder for session {session_id}");

        // Reply with the device's default settings — the client uses these to fill in
        // `RemoteDevice::defaults` so it can resolve op dtypes without an extra RTT.
        let settings = session_manager.device_settings();
        let init_response = TaskResponse {
            content: TaskResponseContent::Init(settings),
            // Zero connection id for handshake; the client doesn't route this through
            // its normal callback map.
            id: ConnectionId::new(0, StreamId::current()),
        };
        let bytes = match rmp_serde::to_vec(&init_response) {
            Ok(b) => b,
            Err(err) => {
                log::error!("Failed to encode Init response: {err:?}");
                return;
            }
        };
        if let Err(err) = socket.send(Message::new(bytes.into())).await {
            log::error!("Failed to send Init response for session {session_id}: {err:?}");
            return;
        }

        let mut receiver = match session_manager.take_response_receiver(session_id).await {
            Ok(r) => r,
            Err(err) => {
                log::error!("{err}");
                return;
            }
        };

        log::info!("Response writer running for session {session_id}");

        // Drain the per-session response queue. The queue is closed when:
        //   (a) all senders are dropped (session closed cleanly), or
        //   (b) the request handler abandons the session on error and `close` is called.
        while let Some(response) = receiver.recv().await {
            let bytes = match rmp_serde::to_vec(&response) {
                Ok(b) => b,
                Err(err) => {
                    log::error!(
                        "Failed to encode response for connection {:?}: {err:?}",
                        response.id
                    );
                    continue;
                }
            };
            if let Err(err) = socket.send(Message::new(bytes.into())).await {
                log::warn!(
                    "Response send failed for session {session_id}: {err:?}; closing writer"
                );
                return;
            }
        }

        log::info!("Response writer for session {session_id} exited (queue closed)");
    }

    async fn handle_socket_request(
        session_manager: Arc<SessionManager<B, P>>,
        mut socket: <P::Server as ProtocolServer>::Channel,
    ) {
        log::info!("[Request Handler] On new connection.");
        let mut session_id: Option<SessionId> = None;

        loop {
            let msg = match socket.recv().await {
                Ok(Some(m)) => m,
                Ok(None) => {
                    log::info!("Request stream closed");
                    break;
                }
                Err(err) => {
                    log::warn!("Request stream error: {err:?}; closing");
                    break;
                }
            };

            let task: Task = match rmp_serde::from_slice(&msg.data) {
                Ok(t) => t,
                Err(err) => {
                    log::error!("Failed to decode request task: {err:?}; closing stream");
                    break;
                }
            };

            match task {
                Task::Init(id) => {
                    log::info!("Init requester for session {id}");
                    session_id = Some(id);
                }
                Task::Close(id) => {
                    log::info!("Close requested for session {id}");
                    Self::handle_close(&session_manager, id).await;
                    return;
                }
                Task::Compute(compute, conn_id) => {
                    let Some(sid) = session_id else {
                        log::error!(
                            "Compute task received before session Init; closing request stream"
                        );
                        break;
                    };

                    if let Err(err) =
                        Self::process_compute(&session_manager, sid, compute, conn_id).await
                    {
                        // A single task failing shouldn't tear down the connection —
                        // the failure surfaces to the client through the response (for
                        // read/sync/dtype tasks) or is logged here (for fire-and-forget
                        // tasks). The connection stays open so subsequent tasks can run.
                        log::error!(
                            "Compute task {conn_id:?} on session {sid} failed: {err}"
                        );
                    }
                }
            }
        }
    }

    async fn handle_close(session_manager: &SessionManager<B, P>, session_id: SessionId) {
        // Ensure backend work for this session is fully drained before we forget the
        // session. `runner.sync()` flushes the session's runner state; `B::sync` flushes
        // the underlying device queue (which the next session may also use, so it's not
        // strictly per-session, but it's cheap and matches the pre-refactor behavior).
        let runner = session_manager.runner(session_id).await;
        if let Err(err) = runner.sync() {
            log::warn!("runner.sync() at session {session_id} close failed: {err:?}");
        }
        let device = runner.device();
        if let Err(err) = B::sync(&device) {
            log::warn!("B::sync(device) at session {session_id} close failed: {err:?}");
        }
        session_manager.close(session_id).await;
    }

    /// Execute a single [`ComputeTask`] inline.
    ///
    /// Sync work is wrapped in [`StreamId::executes`] so the runner's thread-local stream
    /// id matches the one the client assigned to this op. Async work (data-service
    /// transfers, `read_tensor_async`) runs without a stream context — the relevant
    /// stream id is captured into the future at construction time via `executes`.
    async fn process_compute(
        sm: &SessionManager<B, P>,
        session_id: SessionId,
        task: ComputeTask,
        conn_id: ConnectionId,
    ) -> Result<(), String> {
        let runner = sm.runner(session_id).await;
        let runner = &runner;
        let stream_id = conn_id.stream_id;

        match task {
            ComputeTask::RegisterOperations(ops) => {
                stream_id.executes(|| {
                    for op in ops {
                        runner.register_op(op);
                    }
                });
                Ok(())
            }
            ComputeTask::RegisterTensor(id, data) => {
                stream_id.executes(|| runner.register_tensor_data_id(id, data));
                Ok(())
            }
            ComputeTask::RegisterTensorRemote(remote, new_id) => {
                log::info!(
                    "Registering remote tensor (transfer {:?} from {:?})",
                    remote.transfer_id,
                    remote.address,
                );
                let data = sm
                    .data_service
                    .download_tensor(remote.address.clone(), remote.transfer_id)
                    .await
                    .ok_or_else(|| {
                        format!(
                            "Failed to download tensor for transfer {:?} from {:?}",
                            remote.transfer_id, remote.address,
                        )
                    })?;
                stream_id.executes(|| runner.register_tensor_data_id(new_id, data));
                Ok(())
            }
            ComputeTask::ExposeTensorRemote {
                tensor,
                count,
                transfer_id,
            } => {
                log::info!("Exposing tensor (transfer {transfer_id:?})");
                // `read_tensor_async` is sync at construction (it locks handles, fetches
                // the tensor primitive) and returns a future for the actual data read.
                // The stream id matters for the construction step on stream-aware
                // backends.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let data = fut.await.map_err(|e| {
                    format!("read_tensor_async for transfer {transfer_id:?} failed: {e:?}")
                })?;
                sm.data_service.expose_data(data, count, transfer_id).await;
                Ok(())
            }
            ComputeTask::Seed(seed) => {
                stream_id.executes(|| runner.seed(seed));
                Ok(())
            }
            ComputeTask::ReadTensor(tensor) => {
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let data = fut.await;
                Self::send_response(
                    sm,
                    session_id,
                    conn_id,
                    TaskResponseContent::ReadTensor(data),
                )
                .await
            }
            ComputeTask::SyncBackend => {
                let res = stream_id.executes(|| runner.sync());
                Self::send_response(
                    sm,
                    session_id,
                    conn_id,
                    TaskResponseContent::SyncBackend(res),
                )
                .await
            }
            ComputeTask::DTypeUsage(dtype) => {
                let res = stream_id.executes(|| runner.dtype_usage(dtype));
                Self::send_response(
                    sm,
                    session_id,
                    conn_id,
                    TaskResponseContent::DTypeUsage(res),
                )
                .await
            }
        }
    }

    async fn send_response(
        sm: &SessionManager<B, P>,
        session_id: SessionId,
        conn_id: ConnectionId,
        content: TaskResponseContent,
    ) -> Result<(), String> {
        let sender = sm.response_sender(session_id).await;
        sender
            .send(TaskResponse {
                content,
                id: conn_id,
            })
            .await
            .map_err(|_| {
                format!(
                    "Response receiver dropped before result for {conn_id:?} could be sent"
                )
            })
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
