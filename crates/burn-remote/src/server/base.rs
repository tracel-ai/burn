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
use tokio::sync::mpsc;

use crate::shared::{ComputeTask, SessionId, Task, TaskResponse, TaskResponseContent};

use super::session::SessionManager;

/// HTTP-style server for the burn-remote protocol.
///
/// The request-handling task does **no compute**: it decodes the incoming task batch and
/// forwards each [`ComputeTask`] to the session's worker thread (see
/// [`SessionWorker`](super::worker::SessionWorker)) over a bounded channel. Each session owns
/// one worker thread that holds the session's runner and processes its tasks in FIFO order,
/// so a blocking op (e.g. an all-reduce barrier) parks only that session's worker rather than
/// a shared runtime thread. The [`SessionManager`] owns the per-session response queue (one
/// `mpsc::Sender` per session, drained by the response-writing task) and the worker handle.
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
        let thread_id = std::thread::current().id();
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

        log::info!(
            "[{thread_id:?}] Init responder for session {session_id} (device {device_index})"
        );

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
        // session's worker (and its runner) is first created (on the first compute task).
        let mut device_index: u32 = 0;
        // Whether we already ran `handle_close` for this session via an explicit `Task::Close`.
        // If not, we close on the way out so an abrupt disconnect doesn't leak the session.
        let mut session_closed = false;
        // The channel forwarding compute tasks to this session's worker thread, resolved once
        // on the first compute task and reused for the whole connection so we don't re-lock the
        // sessions map per task. Dropped when this handler returns, which — together with the
        // map entry being removed on close — lets the worker's channel close so it can exit.
        let mut task_sender: Option<mpsc::Sender<ComputeTask>> = None;

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
                        // Re-resolve the worker channel for the new session on the next
                        // compute task.
                        task_sender = None;
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

                        if task_sender.is_none() {
                            task_sender =
                                Some(session_manager.session_task_sender(sid, device_index).await);
                        }
                        let sender = task_sender.as_ref().unwrap();

                        // Forward to the session worker. A full channel applies async
                        // backpressure here (we stop reading the socket) rather than blocking
                        // the runtime thread. A send error means the worker is gone, which
                        // shouldn't happen while the session is live — tear the stream down.
                        if sender.send(compute).await.is_err() {
                            log::error!(
                                "Session {sid} worker channel closed; closing request stream"
                            );
                            closed = true;
                            break;
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
                // worker, runner, and response queue aren't leaked: removing the map entry and
                // dropping `task_sender` on return closes the worker's channel, so it flushes
                // and exits, which in turn closes the response writer's queue.
                log::info!("Request stream for session {id} ended without Close; cleaning up");
                Self::handle_close(&session_manager, id).await;
            }
            Some(id) => log::info!("Closing session {id}"),
            None => log::info!("Closing session (no id info)"),
        }
    }

    async fn handle_close(session_manager: &SessionManager<B, P>, session_id: SessionId) {
        // Removing the session drops the worker handle the map holds; once the request
        // connection also drops its cloned task sender (on handler return), the worker's
        // channel closes, so the worker flushes its runner (`runner.sync()` + `B::sync`) and
        // exits on its own thread. A `Close` for an unknown or already-removed session is a
        // no-op `HashMap::remove`, so it never resurrects a phantom session.
        session_manager.close(session_id).await;
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
