use burn_communication::{
    CommunicationChannel, Message, Protocol, ProtocolServer,
    external_comm::{ExternalCommServer, ExternalCommService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::{RouterClient, TensorInterpreter};
use tokio::sync::mpsc;

use crate::shared::{ComputeTask, RequestId, SessionId, Task, TaskResponse, TaskResponseContent};

use super::session::SessionManager;

/// HTTP-style server for the burn-remote protocol.
///
/// Compute tasks are processed **inline** on the request-handling task: there is no
/// per-stream worker thread and no compute-side mpsc channel. The
/// [`SessionManager`] only owns the per-session response queue (one `mpsc::Sender` per
/// session, drained by the response-writing task). The stream id carried on every compute
/// task (ops, tensor registration, reads, syncs, and the same-host/cross-server transfer
/// tasks) is threaded through to the runner via [`StreamId::executes`](burn_std::id::StreamId::executes),
/// so stream-aware backends (fusion, etc.) see the right stream context per op.
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
    /// Start the server hosting the given devices.
    ///
    /// `devices` is indexed by the device index the client selects at session init;
    /// `devices[0]` is the default device. Must be non-empty.
    pub async fn start(devices: Vec<Device<B>>, server: P::Server) {
        let cancel_token = CancellationToken::new();
        let external_comm = Arc::new(ExternalCommService::<B, P>::new(cancel_token));
        let session_manager = Arc::new(SessionManager::<B, P>::new(devices, external_comm.clone()));

        let _server = server
            .route("/response", {
                let session_manager = session_manager.clone();
                move |stream| Self::handle_socket_response(session_manager, stream)
            })
            .route("/request", {
                let session_manager = session_manager.clone();
                move |stream| Self::handle_socket_request(session_manager, stream)
            })
            .route_external_comm(external_comm)
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

        let (session_id, device_index) = match rmp_serde::from_slice::<Vec<Task>>(&msg.data) {
            Ok(mut tasks) => match tasks.pop() {
                Some(Task::Init(id, device_index)) if tasks.is_empty() => (id, device_index),
                other => {
                    log::error!(
                        "Response handshake expected a single Task::Init, got {other:?}; closing stream"
                    );
                    return;
                }
            },
            Err(err) => {
                log::error!("Failed to decode response init handshake: {err:?}");
                return;
            }
        };

        log::info!("Init responder for session {session_id} (device {device_index})");

        // Reply with the selected device's default settings — the client uses these to fill
        // in `RemoteDevice::defaults` so it can resolve op dtypes without an extra RTT.
        let settings = session_manager.device_settings(device_index);
        let device_count = session_manager.device_count();
        let init_response = TaskResponse {
            content: TaskResponseContent::Init(settings, device_count),
            // Placeholder id for the handshake; the client reads this response inline
            // before the response-demux task starts, so it never goes through the
            // pending-callback map.
            id: 0,
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

        let mut receiver = match session_manager
            .take_response_receiver(session_id, device_index)
            .await
        {
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
                        "Failed to encode response for request {:?}: {err:?}",
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
        // The device index this session is bound to, fixed by `Task::Init`. Used when the
        // session's runner is first created (on the first compute task).
        let mut device_index: u32 = 0;
        // Whether we already ran `handle_close` for this session via an explicit `Task::Close`.
        // If not, we close on the way out so an abrupt disconnect doesn't leak the session.
        let mut session_closed = false;
        // The session's runner + response sender, resolved once on the first compute task and
        // reused for the whole connection so we don't re-lock the sessions map per task.
        let mut handles: Option<(TensorInterpreter<B>, mpsc::Sender<TaskResponse>)> = None;

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

            let tasks: Vec<Task> = match rmp_serde::from_slice(&msg.data) {
                Ok(t) => t,
                Err(err) => {
                    log::error!("Failed to decode request batch: {err:?}; closing stream");
                    break;
                }
            };

            // Each websocket frame may carry a batch of tasks (the client buffers
            // fire-and-forget tasks before sending). Process them in order — within a
            // single batch a `Task::Close` short-circuits the rest, which matches the
            // client side: once the client sends `Close`, it sends nothing after.
            let mut closed = false;
            for task in tasks {
                match task {
                    Task::Init(id, index) => {
                        log::info!("Init requester for session {id} (device {index})");
                        session_id = Some(id);
                        device_index = index;
                        // Re-resolve handles for the new session on the next compute task.
                        handles = None;
                    }
                    Task::Close(id) => {
                        log::info!("Close requested for session {id}");
                        Self::handle_close(&session_manager, id).await;
                        session_closed = true;
                        closed = true;
                        break;
                    }
                    Task::Compute(compute) => {
                        let Some(sid) = session_id else {
                            log::error!(
                                "Compute task received before session Init; closing request stream"
                            );
                            closed = true;
                            break;
                        };

                        if handles.is_none() {
                            handles =
                                Some(session_manager.session_handles(sid, device_index).await);
                        }
                        let (runner, sender) = handles.as_ref().unwrap();

                        if let Err(err) =
                            Self::process_compute(&session_manager, runner, sender, compute).await
                        {
                            // A single task failing shouldn't tear down the connection —
                            // the failure surfaces to the client through the response (for
                            // read/sync/dtype tasks) or is logged here (for fire-and-forget
                            // tasks). The connection stays open so subsequent tasks can run.
                            log::error!("Compute task on session {sid} failed: {err}");
                        }
                    }
                }
            }
            if closed {
                break;
            }
        }

        match session_id {
            Some(id) if !session_closed => {
                // The request stream ended without an explicit `Close` (client crash,
                // dropped socket, decode/stream error). Tear the session down here so its
                // runner, handle container, and response queue aren't leaked, and the
                // response writer's queue closes so its task exits too.
                log::info!("Request stream for session {id} ended without Close; cleaning up");
                Self::handle_close(&session_manager, id).await;
            }
            Some(id) => log::info!("Closing session {id}"),
            None => log::info!("Closing session (no id info)"),
        }
    }

    async fn handle_close(session_manager: &SessionManager<B, P>, session_id: SessionId) {
        // A `Close` for an unknown or already-removed session is a no-op: don't resurrect a
        // fresh runner (allocating a new handle container + backend device state) just to
        // sync and immediately drop it.
        let Some(runner) = session_manager.try_runner(session_id).await else {
            return;
        };
        // Ensure backend work for this session is fully drained before we forget the
        // session. `runner.sync()` flushes the session's runner state; `B::sync` flushes
        // the underlying device queue (which the next session may also use, so it's not
        // strictly per-session, but it's cheap and matches the pre-refactor behavior).
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
    /// Sync work is wrapped in [`StreamId::executes`](burn_std::id::StreamId::executes) so the runner's thread-local stream
    /// id matches the one the client assigned to this op. Response-producing tasks carry
    /// their own [`RequestId`] for routing the response back to the right pending
    /// callback on the client. Async work (data-service transfers, `read_tensor_async`)
    /// runs without a stream context — the relevant stream id is captured into the future
    /// at construction time via `executes`.
    async fn process_compute(
        sm: &SessionManager<B, P>,
        runner: &TensorInterpreter<B>,
        sender: &mpsc::Sender<TaskResponse>,
        task: ComputeTask,
    ) -> Result<(), String> {
        match task {
            ComputeTask::RegisterOperation(stream_id, op) => {
                stream_id.executes(|| runner.register_op(op));
                Ok(())
            }
            ComputeTask::RegisterTensor(stream_id, id, data) => {
                stream_id.executes(|| runner.register_tensor_data_id(id, data));
                Ok(())
            }
            ComputeTask::RegisterTensorRemote(stream_id, remote, new_id) => {
                log::info!(
                    "Registering remote tensor (transfer {:?} from {:?})",
                    remote.transfer_id,
                    remote.address,
                );
                let data = sm
                    .external_comm
                    .download_tensor(remote.address.clone(), remote.transfer_id)
                    .await
                    .ok_or_else(|| {
                        format!(
                            "Failed to download tensor for transfer {:?} from {:?}",
                            remote.transfer_id, remote.address,
                        )
                    })?;
                // Register on the client stream that will consume `new_id`, carried over the
                // wire — not the arbitrary tokio worker running this task.
                stream_id.executes(|| runner.register_tensor_data_id(new_id, data));
                Ok(())
            }
            ComputeTask::ExposeTensorLocal {
                stream_id,
                tensor,
                transfer_id,
            } => {
                // Source side of a same-host transfer. Grab the device-resident primitive
                // (no host readback) and park it in the registry for the target session to
                // pick up. Runs inline on this connection, so it is ordered after the op that
                // produced `tensor` — the handle is guaranteed present. Read it back on the
                // client stream that produced it, carried over the wire.
                let kind = stream_id.executes(|| runner.get_tensor(&tensor));
                sm.local_comm.expose(transfer_id, kind).await;
                Ok(())
            }
            ComputeTask::RegisterTensorLocal {
                stream_id,
                transfer_id,
                new_id,
            } => {
                // Target side of a same-host transfer. Wait for the source to expose the
                // primitive, then move it onto this session's device and register it. Awaited
                // inline so subsequent ops on this connection that consume `new_id` see it
                // registered first — same ordering contract as `RegisterTensorRemote`.
                let kind = sm.local_comm.take(transfer_id).await;
                stream_id.executes(|| runner.register_tensor_to_device(new_id, kind));
                Ok(())
            }
            ComputeTask::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                transfer_id,
            } => {
                log::info!("Exposing tensor (transfer {transfer_id:?})");
                // Same shape as `ReadTensor`: the sync part of `read_tensor_async` runs
                // inline to preserve stream ordering, but the readback + expose are
                // detached so a cross-server hand-off doesn't stall this client's op
                // registration on a GPU→host copy. A target that downloads before the
                // expose lands simply blocks on the data service's `new_tensor_notify`,
                // so there is no race.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let external_comm = sm.external_comm.clone();
                tokio::spawn(async move {
                    match fut.await {
                        Ok(data) => {
                            external_comm.expose_data(data, count, transfer_id).await;
                        }
                        Err(e) => {
                            log::error!(
                                "read_tensor_async for transfer {transfer_id:?} failed: {e:?}"
                            );
                        }
                    }
                });
                Ok(())
            }
            ComputeTask::Seed(seed) => {
                runner.seed(seed);
                Ok(())
            }
            ComputeTask::ReadTensor(request_id, stream_id, tensor) => {
                // `read_tensor_async` is sync at construction — it locks the context and
                // captures the tensor's position in the command stream — and returns a
                // future for the actual host readback. Run the sync part inline (so
                // ordering vs. later ops is preserved), then detach the readback await
                // onto its own task. Awaiting it inline would stall the request loop on
                // the GPU→host copy and stop us registering subsequent ops, draining the
                // device queue into a bubble. The client demuxes responses by request id,
                // so out-of-order completion is fine.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let sender = sender.clone();
                tokio::spawn(async move {
                    let data = fut.await;
                    if sender
                        .send(TaskResponse {
                            content: TaskResponseContent::ReadTensor(data),
                            id: request_id,
                        })
                        .await
                        .is_err()
                    {
                        log::warn!(
                            "Response receiver dropped before read for request {request_id} could be sent"
                        );
                    }
                });
                Ok(())
            }
            ComputeTask::SyncBackend(request_id, stream_id) => {
                let res = stream_id.executes(|| runner.sync());
                Self::send_response(sender, request_id, TaskResponseContent::SyncBackend(res)).await
            }
            ComputeTask::DTypeUsage(request_id, dtype) => {
                let res = runner.dtype_usage(dtype);
                Self::send_response(sender, request_id, TaskResponseContent::DTypeUsage(res)).await
            }
        }
    }

    async fn send_response(
        sender: &mpsc::Sender<TaskResponse>,
        request_id: RequestId,
        content: TaskResponseContent,
    ) -> Result<(), String> {
        sender
            .send(TaskResponse {
                content,
                id: request_id,
            })
            .await
            .map_err(|_| {
                format!(
                    "Response receiver dropped before result for request {request_id} could be sent"
                )
            })
    }
}

/// Start the server on the given port, hosting the given [devices](Device).
///
/// `devices` is indexed by the device index the client selects; `devices[0]` is the default.
pub async fn start_websocket_async<B: BackendIr>(devices: Vec<Device<B>>, port: u16) {
    let server = WsServer::new(port);
    RemoteServer::<B, WebSocket>::start(devices, server).await;
}

#[tokio::main]
/// Start the server on the given port, hosting the given [devices](Device).
pub async fn start_websocket<B: BackendIr>(devices: Vec<Device<B>>, port: u16) {
    start_websocket_async::<B>(devices, port).await;
}
